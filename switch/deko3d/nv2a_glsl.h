#pragma once
// GLSL 4.60 for the title's shaders and fixed function state, for UAM (deko3d). The same translation as the SDK's HLSL
// generator (runtime/gpu/shaders.cpp), from the same keys, read the same way.
//
// Interface both stages share:
//   uniform block binding 0: vec4 c[192], the vertex constants in hardware order (D3D register + 96), vertex stage
//   uniform block binding 1: gpu::DrawConstants, std140 (it holds only vec4 arrays and 4x4 matrices)
//   vertex input vN: attribute location N
//   varyings: d0 0, d1 1, b0 2, b1 3, tc0..tc3 4..7, fog amount 8, quad stretch 9
//   fragment textures 0..3: stage 0..3 (sampler2D, sampler3D, samplerCube or sampler2DShadow by TexInfo)
//   fragment output: location 0
#include <cstdint>
#include <string>
#include <vector>

#include "gpu/shaders.h"

namespace gpu::dk {

// raised with every change to the GLSL these functions write: compiled shaders are cached per version
constexpr uint32_t kGeneratorVersion = 3;

// TexInfo flag (bit 0, a backend's own): a target drawn in part this frame (dynamic resolution), reads clamped to it
constexpr uint8_t kTexPart = 1;

// a vertex program, regs the v registers the draw supplies, decode per register as gpu::Assembled::decode
std::string VertexProgramGlsl(const std::vector<uint32_t>& program, uint16_t regs, uint64_t decode, bool quad);
std::string FixedVertexGlsl(uint16_t regs, bool pretransformed, uint64_t decode, const FixedKey& f, bool quad);
// alpha_test false: the draw has the alpha test off, so the shader leaves out its discard (and keeps early depth)
std::string PixelGlsl(const PixelKey& k, bool alpha_test = true);

}  // namespace gpu::dk
