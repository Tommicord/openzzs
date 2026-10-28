/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "mllee/status.hpp"
#include "onnx_hash.hpp"

#include <cstdint>

/*
 * Hand-written ONNX wire primitives.
 *
 * The assembly target is selected at configure time, exactly like the mllee
 * kernels: no CPUID, no dispatch table, no writable module state.  Every
 * routine is a leaf and the portable fallback has identical semantics, so a
 * build without an assembly target still parses the same byte streams.
 */
#if defined(ZZS_MLLEE_ASM_X86_64) || defined(ZZS_MLLEE_ASM_CORTEX_M4)
#define ZZS_MLLEE_ONNX_HAS_ASM 1
extern "C"
{
    /** @brief Decode a 32-bit varint; returns bytes consumed, 0 on failure. */
    uint32_t zzs_mllee_onnx_varint32 (const uint8_t *src, uint32_t available,
                                      uint32_t *out);
    /** @brief Decode a 64-bit varint; returns bytes consumed, 0 on failure. */
    uint32_t zzs_mllee_onnx_varint64 (const uint8_t *src, uint32_t available,
                                      uint64_t *out);
    /** @brief FNV-1a over a byte range. */
    uint32_t zzs_mllee_onnx_hash32 (const uint8_t *src, uint32_t length);
    /** @brief Non-overlapping copy; dst/src are caller-owned. */
    void zzs_mllee_onnx_copy (uint8_t *dst, const uint8_t *src,
                              uint32_t length);
}
#endif

namespace zzs::mllee::onnx
{
/**
 * @brief Portable 32-bit varint decoder with the assembly ABI.
 *
 * Returns the number of consumed bytes, or 0 when the range ends before the
 * terminator or the value does not fit in 32 bits.  The fifth byte may only
 * carry four payload bits; a continuation bit or an upper bit set there is a
 * malformed stream, not a wrap-around.
 */
[[nodiscard]] inline uint32_t
varint32_decode (const uint8_t *const data, const uint32_t available,
                 uint32_t &out) noexcept
{
    uint32_t result = 0;
    uint32_t shift = 0;
    const uint32_t limit = available < 5u ? available : 5u;
    for (uint32_t i = 0; i < limit; ++i)
        {
            const uint8_t byte = data[i];
            if (i == 4u && (byte & 0xF0u) != 0u)
                return 0;
            result |= static_cast<uint32_t> (byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0u)
                {
                    out = result;
                    return i + 1u;
                }
            shift += 7u;
        }
    return 0;
}

/** @brief Portable 64-bit varint decoder with the assembly ABI. */
[[nodiscard]] inline uint32_t
varint64_decode (const uint8_t *const data, const uint32_t available,
                 uint64_t &out) noexcept
{
    uint64_t result = 0;
    uint32_t shift = 0;
    const uint32_t limit = available < 10u ? available : 10u;
    for (uint32_t i = 0; i < limit; ++i)
        {
            const uint8_t byte = data[i];
            if (i == 9u && (byte & 0xFEu) != 0u)
                return 0;
            result |= static_cast<uint64_t> (byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0u)
                {
                    out = result;
                    return i + 1u;
                }
            shift += 7u;
        }
    return 0;
}

/** @brief Decode a 32-bit varint, using the target kernel when available. */
[[nodiscard]] inline uint32_t
varint32_bytes (const uint8_t *const data, const uint32_t available,
                uint32_t &out) noexcept
{
#if defined(ZZS_MLLEE_ONNX_HAS_ASM)
    return zzs_mllee_onnx_varint32 (data, available, &out);
#else
    return varint32_decode (data, available, out);
#endif
}

/** @brief Decode a 64-bit varint, using the target kernel when available. */
[[nodiscard]] inline uint32_t
varint64_bytes (const uint8_t *const data, const uint32_t available,
                uint64_t &out) noexcept
{
#if defined(ZZS_MLLEE_ONNX_HAS_ASM)
    return zzs_mllee_onnx_varint64 (data, available, &out);
#else
    return varint64_decode (data, available, out);
#endif
}

/** @brief FNV-1a over a byte range, dispatched to the target kernel. */
[[nodiscard]] inline uint32_t
hash_bytes (const uint8_t *const data, const uint32_t length) noexcept
{
#if defined(ZZS_MLLEE_ONNX_HAS_ASM)
    return zzs_mllee_onnx_hash32 (data, length);
#else
    return fnv1a (data, length);
#endif
}

/** @brief Second, independent hash used for operator-name collision checks. */
[[nodiscard]] inline uint32_t
hash_bytes2 (const uint8_t *const data, const uint32_t length) noexcept
{
    return djb2 (data, length);
}

/** @brief Non-overlapping byte copy, dispatched to the target kernel. */
inline void
copy_bytes (uint8_t *const dst, const uint8_t *const src,
            const uint32_t length) noexcept
{
#if defined(ZZS_MLLEE_ONNX_HAS_ASM)
    zzs_mllee_onnx_copy (dst, src, length);
#else
    for (uint32_t i = 0; i < length; ++i)
        dst[i] = src[i];
#endif
}
} // namespace zzs::mllee::onnx
