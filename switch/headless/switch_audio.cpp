// Black's mixer (src/sound, kChannels wide at kRate) out through Horizon's audio renderer, as the CoD4 port drives it
// (audrv over audren, 48 kHz stereo final mix, an update thread woken by every renderer frame).
//
// One stereo Int16 voice is fed from a ring of wave buffers of kFrames each. The update thread, on core 0 away from the
// guest's core, waits for each renderer frame (5 ms), and for every buffer the renderer has finished renders the next
// kFrames from the mixer, folds the 5.1 mix to stereo as the XAudio2 output does (L = FL + 0.707 C + 0.707 BL, R likewise,
// no sub; mono: the average of the two), clips and converts it, and queues it. kQueued buffers stay queued, about
// 30 ms ahead. Paused (focus lost), the voice pauses and the mixer is not run.
// Every 10 s the log has the mixer's time per buffer and the underruns (the queue found empty).
#include <switch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <malloc.h>
#include <vector>

#include "audio/output.h"
#include "core/log.h"

namespace audio {

namespace {

constexpr uint32_t kFrames = kRate / 200;  // 5 ms, one renderer frame
constexpr int kBuffers = 8;
constexpr int kQueued = 6;
constexpr float kCentreFold = 0.707f;
constexpr float kBackFold = 0.707f;

std::atomic<uint32_t> g_mode{kOutputAuto};
std::atomic<bool> g_paused{false};
std::atomic<bool> g_stop{false};

const AudioRendererConfig kConfig = {
    AudioRendererOutputRate_48kHz,  // output_rate
    4,                              // num_voices (a stereo voice takes two)
    0,                              // num_effects
    1,                              // num_sinks
    1,                              // num_mix_objs
    2,                              // num_mix_buffers
};

struct Output {
    AudioDriver drv{};
    AudioDriverWaveBuf bufs[kBuffers]{};
    int16_t* pcm = nullptr;  // kBuffers * kFrames stereo frames, in the mempool
    std::function<void(float*, uint32_t)> render;
    std::vector<float> mix;
    Thread thread{};
    // statistics since the last report
    uint64_t render_ticks = 0, render_max = 0, rendered = 0, underruns = 0, queued_min = kBuffers;
    uint64_t report_at = 0;
};
Output* g_out = nullptr;

inline int16_t ToPcm(float s) { return int16_t(std::lrintf(std::clamp(s, -1.0f, 1.0f) * 32767.0f)); }

void Fold(const float* in, int16_t* out, uint32_t frames, bool mono) {
    for (uint32_t i = 0; i < frames; i++, in += kChannels) {
        float c = in[Centre] * kCentreFold;
        float l = in[FrontLeft] + c + in[BackLeft] * kBackFold;
        float r = in[FrontRight] + c + in[BackRight] * kBackFold;
        if (mono) l = r = (l + r) * 0.5f;
        out[2 * i] = ToPcm(l);
        out[2 * i + 1] = ToPcm(r);
    }
}

void Fill(Output& o, int i) {
    int16_t* pcm = o.pcm + size_t(i) * kFrames * 2;
    const uint64_t t0 = armGetSystemTick();
    o.render(o.mix.data(), kFrames);
    Fold(o.mix.data(), pcm, kFrames, g_mode == kOutputMono);
    const uint64_t spent = armGetSystemTick() - t0;
    o.render_ticks += spent;
    o.render_max = std::max(o.render_max, spent);
    o.rendered++;
    armDCacheFlush(pcm, kFrames * 2 * sizeof(int16_t));
    AudioDriverWaveBuf& b = o.bufs[i];
    b.data_pcm16 = pcm;
    b.size = kFrames * 2 * sizeof(int16_t);
    b.start_sample_offset = 0;
    b.end_sample_offset = kFrames;
    b.is_looping = false;
    audrvVoiceAddWaveBuf(&o.drv, 0, &b);
}

void Report(Output& o) {
    const uint64_t now = armGetSystemTick();
    if (!o.report_at) o.report_at = now;
    if (armTicksToNs(now - o.report_at) < 10'000'000'000ull) return;
    double ms = o.rendered ? double(armTicksToNs(o.render_ticks)) / 1e6 / double(o.rendered) : 0.0;
    double share = double(armTicksToNs(o.render_ticks)) / double(armTicksToNs(now - o.report_at)) * 100.0;
    LOGI("audio: %llu buffers of 5 ms, mixer %.3f ms each (max %.3f, %.1f%% of a core), %llu underruns, queue low %llu, "
         "%u voice drops",
         (unsigned long long)o.rendered, ms, double(armTicksToNs(o.render_max)) / 1e6, share,
         (unsigned long long)o.underruns, (unsigned long long)o.queued_min, audrvVoiceGetVoiceDropsCount(&o.drv, 0));
    o.render_ticks = o.render_max = o.rendered = o.underruns = 0;
    o.queued_min = kBuffers;
    o.report_at = now;
}

void Run(void* arg) {
    Output& o = *static_cast<Output*>(arg);
    bool paused = false;
    while (!g_stop) {
        audrenWaitFrame();
        if (g_paused != paused) {
            paused = g_paused;
            audrvVoiceSetPaused(&o.drv, 0, paused);
            LOGI("audio: %s", paused ? "paused" : "resumed");
        }
        audrvUpdate(&o.drv);
        if (!paused) {
            int queued = 0;
            for (auto& b : o.bufs)
                if (b.state == AudioDriverWaveBufState_Queued || b.state == AudioDriverWaveBufState_Playing ||
                    b.state == AudioDriverWaveBufState_Waiting)
                    queued++;
            o.queued_min = std::min<uint64_t>(o.queued_min, uint64_t(queued));
            if (queued == 0) o.underruns++;
            // the buffers in ring order from the one after the newest queued
            for (int n = 0; n < kBuffers && queued < kQueued; n++) {
                static int next = 0;
                AudioDriverWaveBuf& b = o.bufs[next];
                if (b.state != AudioDriverWaveBufState_Free && b.state != AudioDriverWaveBufState_Done) break;
                Fill(o, next);
                next = (next + 1) % kBuffers;
                queued++;
            }
            audrvUpdate(&o.drv);
            if (!audrvVoiceIsPlaying(&o.drv, 0)) audrvVoiceStart(&o.drv, 0);
        }
        Report(o);
    }
}

bool Open(Output& o) {
    Result rc = audrenInitialize(&kConfig);
    if (R_FAILED(rc)) {
        LOGE("audio: audrenInitialize failed (%X), no sound", rc);
        return false;
    }
    rc = audrvCreate(&o.drv, &kConfig, 2);
    if (R_FAILED(rc)) {
        LOGE("audio: audrvCreate failed (%X), no sound", rc);
        audrenExit();
        return false;
    }
    const size_t bytes = (size_t(kBuffers) * kFrames * 2 * sizeof(int16_t) + 0xFFF) & ~size_t(0xFFF);
    o.pcm = static_cast<int16_t*>(memalign(0x1000, bytes));
    if (!o.pcm) {
        LOGE("audio: no memory for the wave buffers, no sound");
        audrvClose(&o.drv);
        audrenExit();
        return false;
    }
    std::memset(o.pcm, 0, bytes);
    armDCacheFlush(o.pcm, bytes);
    int pool = audrvMemPoolAdd(&o.drv, o.pcm, bytes);
    static const u8 kSink[] = {0, 1};
    if (pool < 0 || !audrvMemPoolAttach(&o.drv, pool) || audrvDeviceSinkAdd(&o.drv, AUDREN_DEFAULT_DEVICE_NAME, 2, kSink) < 0 ||
        !audrvVoiceInit(&o.drv, 0, 2, PcmFormat_Int16, kRate)) {
        LOGE("audio: the audio renderer refused the voice, no sound");
        audrvClose(&o.drv);
        audrenExit();
        std::free(o.pcm);
        o.pcm = nullptr;
        return false;
    }
    audrvVoiceSetDestinationMix(&o.drv, 0, AUDREN_FINAL_MIX_ID);
    audrvVoiceSetMixFactor(&o.drv, 0, 1.0f, 0, 0);
    audrvVoiceSetMixFactor(&o.drv, 0, 0.0f, 0, 1);
    audrvVoiceSetMixFactor(&o.drv, 0, 0.0f, 1, 0);
    audrvVoiceSetMixFactor(&o.drv, 0, 1.0f, 1, 1);
    rc = audrvUpdate(&o.drv);
    if (R_SUCCEEDED(rc)) rc = audrenStartAudioRenderer();
    if (R_FAILED(rc)) {
        LOGE("audio: the audio renderer did not start (%X), no sound", rc);
        audrvClose(&o.drv);
        audrenExit();
        std::free(o.pcm);
        o.pcm = nullptr;
        return false;
    }
    return true;
}

}  // namespace

bool IsSurround() { return false; }  // the console mixes to stereo
bool IsMono() { return g_mode == kOutputMono; }
void SetOutputMode(uint32_t mode) { g_mode = mode; }
void SetPaused(bool paused) { g_paused = paused; }

void StartOutput(std::function<void(float*, uint32_t)> render) {
    if (g_out) return;
    auto* o = new Output;
    o->render = std::move(render);
    o->mix.assign(size_t(kFrames) * kChannels, 0.0f);
    if (!Open(*o)) {
        delete o;
        return;
    }
    // core 0 (the guest's threads are on core 1, the renderer's on 2), above the runtime's own threads: a late
    // update leaves the voice without a buffer
    Result rc = threadCreate(&o->thread, Run, o, nullptr, 0x10000, 0x24, 0);
    const bool created = R_SUCCEEDED(rc);
    if (created) rc = threadStart(&o->thread);
    if (R_FAILED(rc)) {
        LOGE("audio: no output thread (%X), no sound", rc);
        if (created) threadClose(&o->thread);
        audrenStopAudioRenderer();
        audrvClose(&o->drv);
        audrenExit();
        std::free(o->pcm);
        delete o;
        return;
    }
    g_out = o;
    LOGI("audio: audren 48 kHz stereo, %u buffers of %u frames, %u queued, revision %X", unsigned(kBuffers),
         unsigned(kFrames), unsigned(kQueued), audrenGetRevision());
}


// the clean exit (clean_exit.cpp), before the guest threads pause: the thread ends (its next 5 ms frame) and the
// renderer, its memory pool and the service session close, so the heap pages the service mapped are free again
void StopOutput() {
    Output* o = g_out;
    if (!o) return;
    g_stop = true;
    if (o->thread.handle) {
        threadWaitForExit(&o->thread);
        threadClose(&o->thread);
    }
    audrvVoiceStop(&o->drv, 0);
    audrvUpdate(&o->drv);
    audrvClose(&o->drv);
    audrenExit();
    g_out = nullptr;
}

}  // namespace audio
