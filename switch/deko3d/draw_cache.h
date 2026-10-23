#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace gpu::dk {
// Cache the effective command arguments, not guest state: target-derived values
// and shader variants must participate too. A new recording or auxiliary pass
// advances the epoch, forcing every affected binding to be emitted again.
template<class T> class DrawCache {
    static_assert(std::is_trivially_copyable_v<T>);
    T value_{};
    uint64_t epoch_ = 0;
public:
    bool Update(const T& value, uint64_t epoch, bool enabled) {
        if (!enabled) return true;
        if (epoch_ == epoch && !std::memcmp(&value_, &value, sizeof(T))) return false;
        std::memcpy(&value_, &value, sizeof(T));
        epoch_ = epoch;
        return true;
    }
};
} // namespace gpu::dk
