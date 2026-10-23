#include "shader_cache.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <compiler_iface.h>

#include "core/log.h"
#include "gpu/shaders.h"

namespace gpu {

namespace {

constexpr uint32_t kCodeBlock = 4u << 20;

// DKSH: header, then the control section (control_sz bytes from the start), then the code (code_sz bytes)
struct DkshHeader {
    uint32_t magic, header_sz, control_sz, code_sz, programs_off, num_programs;
};

std::string PathOf(const std::string& dir, const char* prefix, uint64_t h, const char* ext) {
    char name[64];
    std::snprintf(name, sizeof(name), "/%s%016llx%s", prefix, (unsigned long long)h, ext);
    return dir + name;
}

bool ReadFile(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::setvbuf(f, nullptr, _IONBF, 0);  // one fs request for the file, not one per 1 KB buffer
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? size_t(n) : 0);
    bool ok = n > 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

void WriteFile(const std::string& path, const void* p, size_t n) {
    std::string tmp = path + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "wb")) {
        bool ok = std::fwrite(p, 1, n, f) == n;
        std::fclose(f);
        // Horizon does not rename onto an existing file: one there was rejected on load (cut short or stale)
        if (ok) {
            std::remove(path.c_str());
            ok = std::rename(tmp.c_str(), path.c_str()) == 0;
        }
        if (!ok) std::remove(tmp.c_str());
    }
}

}  // namespace

bool CompileDksh(ShaderCache::Stage stage, const std::string& glsl, std::vector<uint8_t>& dksh) {
    DekoCompiler c(stage == ShaderCache::kVertex ? pipeline_stage_vertex : pipeline_stage_fragment);
    if (!c.CompileGlsl(glsl.c_str())) return false;
    void* data = nullptr;
    uint32_t size = 0;
    if (!c.OutputDkshToMemory(&data, &size)) return false;
    dksh.assign(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
    std::free(data);
    return size >= sizeof(DkshHeader) && !std::memcmp(dksh.data(), "DKSH", 4);
}

// files of a directory, removed (the directory too when sub)
static void RemoveFiles(const std::string& dir, bool sub) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::string> names;
    while (dirent* e = readdir(d))
        if (e->d_type != DT_DIR) names.push_back(e->d_name);
    closedir(d);
    for (const std::string& n : names) std::remove((dir + "/" + n).c_str());
    if (sub) rmdir(dir.c_str());
}

void ShaderCache::Init(DkDevice dev, const char* dir, uint32_t version) {
    dev_ = dev;
    // one folder per generator version: a new generator starts empty and the old folders (and the flat files of the
    // first layout) are removed, so the preload only reads shaders this build can ask for
    mkdir(dir, 0777);
    std::string own = "g" + std::to_string(version);
    std::vector<std::string> stale;
    if (DIR* d = opendir(dir)) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n == "." || n == ".." || n == own) continue;
            if (e->d_type == DT_DIR && n[0] == 'g') stale.push_back(n);
            else if (e->d_type != DT_DIR && (n.size() > 5 && (n.compare(n.size() - 5, 5, ".dksh") == 0 || n.compare(0, 5, "fail_") == 0)))
                std::remove((std::string(dir) + "/" + n).c_str());
        }
        closedir(d);
    }
    for (const std::string& n : stale) RemoveFiles(std::string(dir) + "/" + n, true);
    dir_ = std::string(dir) + "/" + own;
    mkdir(dir_.c_str(), 0777);
    Preload();
}

// every DKSH an earlier run left, loaded now: a shader first used mid level then costs no file read (a cold read
// or compile there was a 30 to 176 ms hitch)
void ShaderCache::Preload() {
    DIR* d = opendir(dir_.c_str());
    if (!d) return;
    std::vector<uint8_t> dksh;
    uint32_t n = 0, bad = 0;
    while (dirent* e = readdir(d)) {
        const char* name = e->d_name;
        if (std::strlen(name) != 21 || std::strcmp(name + 16, ".dksh")) continue;
        char* end = nullptr;
        uint64_t h = std::strtoull(std::string(name, 16).c_str(), &end, 16);
        if (shaders_.count(h)) continue;
        auto sh = std::make_unique<DkShader>();
        if (ReadFile(dir_ + "/" + name, dksh) && Load(dksh, *sh)) {
            shaders_[h] = std::move(sh);
            n++;
        } else {
            bad++;
        }
    }
    closedir(d);
    loaded_ += n;
    LOGI("deko3d: %u cached shaders preloaded (%u unreadable)", n, bad);
}

bool ShaderCache::Load(const std::vector<uint8_t>& dksh, DkShader& out) {
    DkshHeader h;
    if (dksh.size() < sizeof(h)) return false;
    std::memcpy(&h, dksh.data(), sizeof(h));
    if (std::memcmp(&h.magic, "DKSH", 4) || h.control_sz + uint64_t(h.code_sz) > dksh.size()) return false;
    uint32_t at = (used_ + DK_SHADER_CODE_ALIGNMENT - 1) & ~(DK_SHADER_CODE_ALIGNMENT - 1);
    if (blocks_.empty() || at + h.code_sz + DK_SHADER_CODE_UNUSABLE_SIZE > size_) {
        uint32_t bytes = std::max<uint32_t>(kCodeBlock, (h.code_sz + DK_SHADER_CODE_UNUSABLE_SIZE + 0xFFF) & ~0xFFFu);
        DkMemBlockMaker m;
        dkMemBlockMakerDefaults(&m, dev_, bytes);
        m.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached | DkMemBlockFlags_Code;
        DkMemBlock b = dkMemBlockCreate(&m);
        if (!b) return false;
        blocks_.push_back(b);
        size_ = bytes;
        at = 0;
    }
    DkMemBlock b = blocks_.back();
    std::memcpy(static_cast<uint8_t*>(dkMemBlockGetCpuAddr(b)) + at, dksh.data() + h.control_sz, h.code_sz);
    used_ = at + h.code_sz;
    DkShaderMaker sm;
    dkShaderMakerDefaults(&sm, b, at);
    sm.control = dksh.data();
    sm.programId = 0;
    dkShaderInitialize(&out, &sm);
    return dkShaderIsValid(&out);
}

const DkShader* ShaderCache::Get(Stage stage, const std::string& glsl) {
    uint64_t h = Hash(glsl.data(), glsl.size(), stage);
    auto it = shaders_.find(h);
    if (it != shaders_.end()) return it->second.get();
    auto sh = std::make_unique<DkShader>();
    std::vector<uint8_t> dksh;
    std::string path = PathOf(dir_, "", h, ".dksh");
    bool ok = false;
    if (ReadFile(path, dksh) && Load(dksh, *sh)) {
        loaded_++;
        ok = true;
    } else if (CompileDksh(stage, glsl, dksh) && Load(dksh, *sh)) {
        compiled_++;
        WriteFile(path, dksh.data(), dksh.size());
        ok = true;
    } else {
        std::string fail = PathOf(dir_, "fail_", h, ".glsl");
        WriteFile(fail, glsl.data(), glsl.size());
        LOGE("deko3d: shader %016llx does not compile, saved as %s", (unsigned long long)h, fail.c_str());
    }
    DkShader* p = ok ? sh.get() : nullptr;
    shaders_[h] = ok ? std::move(sh) : nullptr;
    return p;
}

}  // namespace gpu
