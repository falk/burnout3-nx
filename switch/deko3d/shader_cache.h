#pragma once
// GLSL to loaded deko3d shaders: in memory by the source's hash, then the DKSH file <dir>/<hash>.dksh, then UAM.
// A compile writes its DKSH file, so a run only compiles what no earlier run (or a host prebake) has.
#include <deko3d.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu {

class ShaderCache {
public:
    enum Stage : uint8_t { kVertex, kFragment };
    // dir/g<version>: version names the GLSL generator (dk::kGeneratorVersion), its shaders kept apart
    void Init(DkDevice dev, const char* dir, uint32_t version);
    // null when it does not compile (the source is saved as <dir>/fail_<hash>.glsl)
    const DkShader* Get(Stage stage, const std::string& glsl);
    uint32_t Compiled() const { return compiled_; }
    uint32_t Loaded() const { return loaded_; }

private:
    bool Load(const std::vector<uint8_t>& dksh, DkShader& out);
    void Preload();
    DkDevice dev_ = nullptr;
    std::string dir_;
    std::vector<DkMemBlock> blocks_;
    uint32_t used_ = 0, size_ = 0;
    std::unordered_map<uint64_t, std::unique_ptr<DkShader>> shaders_;
    uint32_t compiled_ = 0, loaded_ = 0;
};

// UAM, GLSL 4.60 to a DKSH image; false with the compiler's message on stderr
bool CompileDksh(ShaderCache::Stage stage, const std::string& glsl, std::vector<uint8_t>& dksh);

}  // namespace gpu
