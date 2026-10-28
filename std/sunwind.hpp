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

#include "std/string_view.hpp"
#include "std/maybe.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace zzs::base
{
class FrameUnwindError
{
  public:
    enum class Kind
    {
        INVALID_ADDR
    };

    FrameUnwindError () = delete;
    explicit constexpr FrameUnwindError (const Kind kind) noexcept
        : _M_kind (kind)
    {
    }
    [[nodiscard]] constexpr Kind
    kind () const noexcept
    {
        return this->_M_kind;
    }
    [[nodiscard]] constexpr bool
    operator== (const FrameUnwindError &other) const noexcept
    {
        return this->_M_kind == other._M_kind;
    }

  private:
    Kind _M_kind;
};
inline constexpr size_t _S_unwind_max_frames = 256;
struct UnwindFrame
{
    uintptr_t _M_ip = 0;
    uintptr_t _M_sp = 0;
    Maybe<uintptr_t, FrameUnwindError> _M_module_base{};

    [[nodiscard]]
    constexpr uintptr_t
    symbol_address () const noexcept
    {
        return this->_M_ip;
    }
};
static std::atomic unwind_guard_state{0};

inline bool
unwind_try_enter () noexcept
{
    int expected = 0;
    return unwind_guard_state.compare_exchange_strong (
        expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
}

inline void
unwind_leave () noexcept
{
    unwind_guard_state.store (0, std::memory_order_release);
}

inline bool
unwind_in_handler () noexcept
{
    return unwind_guard_state.load (std::memory_order_acquire) != 0;
}

inline constexpr size_t _S_unwind_max_cached_modules = 64;
inline constexpr size_t unwind_cached_name_size = 128;

struct UnwindModuleEntry
{
    uintptr_t _M_base = 0;
    uintptr_t _M_end = 0;
    char _M_name[unwind_cached_name_size] = {};
};

static UnwindModuleEntry unwind_module_cache[_S_unwind_max_cached_modules] = {};
static std::atomic<size_t> unwind_module_count{0};
static std::atomic unwind_modules_ready{false};

static const UnwindModuleEntry *
unwind_cache_lookup (uintptr_t ip) noexcept
{
    const size_t count = unwind_module_count.load (std::memory_order_acquire);
    if (count == 0)
        return nullptr;
    size_t lo = 0;
    size_t hi = count;
    while (lo < hi)
        {
            const size_t mid = lo + (hi - lo) / 2;
            const UnwindModuleEntry &entry = unwind_module_cache[mid];
            if (ip < entry._M_base)
                hi = mid;
            else if (ip >= entry._M_end)
                lo = mid + 1;
            else
                return &entry;
        }
    return nullptr;
}

static void
unwind_cache_store (size_t index, uintptr_t base, uintptr_t end,
                    const char *name) noexcept
{
    UnwindModuleEntry &entry = unwind_module_cache[index];
    entry._M_base = base;
    entry._M_end = end;
    entry._M_name[0] = '\0';
    if (name == nullptr)
        return;
    size_t i = 0;
    while (i + 1 < unwind_cached_name_size && name[i] != '\0')
        {
            entry._M_name[i] = name[i];
            ++i;
        }
    entry._M_name[i] = '\0';
}

static void
unwind_cache_sort (size_t count) noexcept
{
    for (size_t i = 1; i < count; ++i)
        {
            const UnwindModuleEntry key = unwind_module_cache[i];
            size_t j = i;
            while (j > 0 && unwind_module_cache[j - 1]._M_base > key._M_base)
                {
                    unwind_module_cache[j] = unwind_module_cache[j - 1];
                    --j;
                }
            unwind_module_cache[j] = key;
        }
}

static bool
unwind_is_readable (uintptr_t addr) noexcept
{
    if (addr < 4096)
        return false;
#if defined(ZZS_UNWIND_MINCORE)
    const uintptr_t page = addr & ~uintptr_t{0xFFF};
    unsigned char vec = 0;
#if defined(__linux__)
    if (::mincore (reinterpret_cast<void *> (page), size_t{4096}, &vec) != 0)
        return false;
#else
    if (::mincore (reinterpret_cast<caddr_t> (page), size_t{4096},
                   reinterpret_cast<char *> (&vec))
        != 0)
        return false;
#endif
    return (vec & 0x1u) != 0;
#elif defined(_WIN32)
    MEMORY_BASIC_INFORMATION info = {};
    if (::VirtualQuery (reinterpret_cast<const void *> (addr), &info,
                        sizeof (info))
        == 0)
        return false;
    constexpr DWORD mem_commit = 0x1000;
    constexpr DWORD page_noaccess = 0x01;
    constexpr DWORD page_guard = 0x100;
    return info.State == mem_commit
           && (info.Protect & (page_noaccess | page_guard)) == 0;
#else
    (void) addr;
    return false;
#endif
}

static bool
unwind_read_word (uintptr_t addr, uintptr_t &out) noexcept
{
    if (addr < 4096 || (addr % sizeof (uintptr_t)) != 0)
        return false;
    /* Never straddle a page boundary: only one page is probed. */
    if ((addr & 0xFFF) > 4096 - sizeof (uintptr_t))
        return false;
    if (!unwind_is_readable (addr))
        return false;
    const auto *bytes = reinterpret_cast<const unsigned char *> (addr);
    uintptr_t value = 0;
    for (size_t i = 0; i < sizeof (uintptr_t); ++i)
        value |= uintptr_t{bytes[i]} << (8 * i);
    out = value;
    return true;
}

static uint16_t
unwind_le16 (const uint8_t *p) noexcept
{
    return uint16_t (uint16_t (p[0]) | uint16_t (uint16_t (p[1]) << 8));
}

static uint32_t
unwind_le32 (const uint8_t *p) noexcept
{
    return uint32_t (p[0]) | (uint32_t (p[1]) << 8) | (uint32_t (p[2]) << 16)
           | (uint32_t (p[3]) << 24);
}

static uint64_t
unwind_le64 (const uint8_t *p) noexcept
{
    return uint64_t (unwind_le32 (p)) | (uint64_t (unwind_le32 (p + 4)) << 32);
}
inline constexpr size_t unwind_register_count = 64;

struct UnwindRegs
{
    uintptr_t _M_gpr[unwind_register_count] = {};
    uintptr_t _M_ip = 0;
};

struct UnwindState
{
    UnwindRegs _M_regs = {};
    uintptr_t _M_sp = 0;
    bool _M_stop = false;
};

/* DWARF register index of the stack pointer. */
constexpr size_t
unwind_sp_index () noexcept
{
#if defined(__aarch64__)
    return 31;
#elif defined(__x86_64__)
    return 7;
#elif defined(__i386__)
    return 4;
#elif defined(__arm__)
    return 13;
#else
    return 0;
#endif
}

/* DWARF register index of the frame pointer. */
constexpr size_t
unwind_fp_index () noexcept
{
#if defined(__aarch64__)
    return 29;
#elif defined(__x86_64__)
    return 6;
#else
    return 0;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
static bool
unwind_capture_arch (UnwindRegs &regs) noexcept
{
#if defined(__x86_64__)
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    asm volatile ("leaq 0(%%rip), %0\n\t"
                  "movq %%rsp, %1\n\t"
                  "movq %%rbp, (%2)\n\t"
                  "movq %%rbx, (%3)\n\t"
                  "movq %%r12, (%4)\n\t"
                  "movq %%r13, (%5)\n\t"
                  "movq %%r14, (%6)\n\t"
                  "movq %%r15, (%7)\n\t"
                  : "=&r"(ip), "=&r"(sp)
                  : "r"(&regs._M_gpr[6]), "r"(&regs._M_gpr[3]),
                    "r"(&regs._M_gpr[12]), "r"(&regs._M_gpr[13]),
                    "r"(&regs._M_gpr[14]), "r"(&regs._M_gpr[15])
                  : "memory");
    regs._M_gpr[7] = sp;
    regs._M_gpr[16] = ip;
    regs._M_ip = ip;
    return true;
#elif defined(__aarch64__)
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    asm volatile (
        "adr %0, 1f\n\t"
        "b 2f\n\t"
        "1:\n\t"
        "2:\n\t"
        "mov %1, sp\n\t"
        "str x29, [%2]\n\t"
        "str x30, [%3]\n\t"
        "str x19, [%4]\n\t"
        "str x20, [%5]\n\t"
        "str x21, [%6]\n\t"
        "str x22, [%7]\n\t"
        "str x23, [%8]\n\t"
        "str x24, [%9]\n\t"
        "str x25, [%10]\n\t"
        "str x26, [%11]\n\t"
        "str x27, [%12]\n\t"
        "str x28, [%13]\n\t"
        : "=&r"(ip), "=&r"(sp)
        : "r"(&regs._M_gpr[29]), "r"(&regs._M_gpr[30]), "r"(&regs._M_gpr[19]),
          "r"(&regs._M_gpr[20]), "r"(&regs._M_gpr[21]), "r"(&regs._M_gpr[22]),
          "r"(&regs._M_gpr[23]), "r"(&regs._M_gpr[24]), "r"(&regs._M_gpr[25]),
          "r"(&regs._M_gpr[26]), "r"(&regs._M_gpr[27]), "r"(&regs._M_gpr[28])
        : "memory");
    regs._M_gpr[31] = sp;
    regs._M_ip = ip;
    return true;
#elif defined(__i386__)
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    asm volatile ("movl %%esp, %0\n\t"
                  "movl %%ebp, (%2)\n\t"
                  "movl %%ebx, (%3)\n\t"
                  "movl %%esi, (%4)\n\t"
                  "movl %%edi, (%5)\n\t"
                  "call 2f\n\t"
                  "2:\n\t"
                  "popl %1\n\t"
                  : "=&r"(sp), "=&r"(ip)
                  : "r"(&regs._M_gpr[5]), "r"(&regs._M_gpr[3]),
                    "r"(&regs._M_gpr[6]), "r"(&regs._M_gpr[7])
                  : "memory", "cc");
    regs._M_gpr[4] = sp;
    regs._M_gpr[8] = ip;
    regs._M_ip = ip;
    return true;
#elif defined(__arm__)
    uintptr_t ip = 0;
    uintptr_t sp = 0;
    asm volatile (
        "adr %0, 1f\n\t"
        "1:\n\t"
        "mov %1, sp\n\t"
        "str r11, [%2]\n\t"
        "str r14, [%3]\n\t"
        "str r4, [%4]\n\t"
        "str r5, [%5]\n\t"
        "str r6, [%6]\n\t"
        "str r7, [%7]\n\t"
        "str r8, [%8]\n\t"
        "str r9, [%9]\n\t"
        "str r10, [%10]\n\t"
        : "=&r"(ip), "=&r"(sp)
        : "r"(&regs._M_gpr[11]), "r"(&regs._M_gpr[14]), "r"(&regs._M_gpr[4]),
          "r"(&regs._M_gpr[5]), "r"(&regs._M_gpr[6]), "r"(&regs._M_gpr[7]),
          "r"(&regs._M_gpr[8]), "r"(&regs._M_gpr[9]), "r"(&regs._M_gpr[10])
        : "memory");
    regs._M_gpr[13] = sp;
    regs._M_ip = ip;
    return true;
#else
    (void) regs;
    return false;
#endif
}
#else
static bool
unwind_capture_arch (UnwindRegs &regs) noexcept
{
#if defined(ZZS_UNWIND_RTL)
    ::CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    ::RtlCaptureContext (&ctx);
#if defined(_M_X64) || defined(__x86_64__)
    regs.gpr[0] = uintptr_t (ctx.Rax);
    regs.gpr[1] = uintptr_t (ctx.Rdx);
    regs.gpr[2] = uintptr_t (ctx.Rcx);
    regs.gpr[3] = uintptr_t (ctx.Rbx);
    regs.gpr[4] = uintptr_t (ctx.Rsi);
    regs.gpr[5] = uintptr_t (ctx.Rdi);
    regs.gpr[6] = uintptr_t (ctx.Rbp);
    regs.gpr[7] = uintptr_t (ctx.Rsp);
    regs.gpr[8] = uintptr_t (ctx.R8);
    regs.gpr[9] = uintptr_t (ctx.R9);
    regs.gpr[10] = uintptr_t (ctx.R10);
    regs.gpr[11] = uintptr_t (ctx.R11);
    regs.gpr[12] = uintptr_t (ctx.R12);
    regs.gpr[13] = uintptr_t (ctx.R13);
    regs.gpr[14] = uintptr_t (ctx.R14);
    regs.gpr[15] = uintptr_t (ctx.R15);
    regs.gpr[16] = uintptr_t (ctx.Rip);
    regs.ip = uintptr_t (ctx.Rip);
    return true;
#elif defined(_M_ARM64) || defined(__aarch64__)
    for (size_t i = 0; i <= 28; ++i)
        regs.gpr[i] = uintptr_t (ctx.X[i]);
    regs.gpr[29] = uintptr_t (ctx.Fp);
    regs.gpr[30] = uintptr_t (ctx.Lr);
    regs.gpr[31] = uintptr_t (ctx.Sp);
    regs.ip = uintptr_t (ctx.Pc);
    return true;
#else
    (void) ctx;
    return false;
#endif
#else
    (void) regs;
    return false;
#endif
}
#endif

__attribute__ ((noinline)) bool
unwind_capture_current (UnwindState &state) noexcept
{
    state = UnwindState{};
    if (!unwind_capture_arch (state._M_regs))
        return false;
    /*
     * unwind_capture_arch's own frame is already gone when the walk runs
     * (the stepping machinery reuses that stack region), so retarget the
     * capture to this function's caller: unwind_walk_impl's frame stays
     * alive for the whole walk and everything above it is untouched.
     */
    const uintptr_t caller_fp = uintptr_t (__builtin_frame_address (1));
    state._M_regs._M_ip = uintptr_t (__builtin_return_address (0));
    state._M_regs._M_gpr[unwind_fp_index ()] = caller_fp;
    state._M_regs._M_gpr[unwind_sp_index ()] = caller_fp;
    state._M_sp = caller_fp;
    return state._M_regs._M_ip != 0;
}

static bool
unwind_fp_step (UnwindState &state) noexcept
{
#if defined(__x86_64__) || defined(__aarch64__)
    constexpr size_t fp_index = unwind_fp_index ();
    const uintptr_t fp = state._M_regs._M_gpr[fp_index];
    const uintptr_t sp = state._M_sp;
    if (fp == 0 || (fp % 16) != 0)
        return false;
    if (fp < sp)
        return false;
    uintptr_t next_fp = 0;
    uintptr_t next_ip = 0;
    if (!unwind_read_word (fp, next_fp))
        return false;
    if (!unwind_read_word (fp + 8, next_ip))
        return false;
    const uintptr_t next_sp = fp + 16;
    if (next_ip == 0 || next_sp <= sp)
        return false;
    if (next_fp != 0 && next_fp <= fp)
        return false;
    state._M_regs._M_gpr[fp_index] = next_fp;
    state._M_regs._M_ip = next_ip;
    state._M_sp = next_sp;
    return true;
#else
    (void) state;
    return false;
#endif
}
inline constexpr size_t unwind_state_stack_limit = 32;
inline constexpr size_t unwind_expr_stack_limit = 64;
inline constexpr size_t unwind_max_entry_len = size_t{1} << 28;

enum class UnwindRegRuleKind : unsigned char
{
    sameValue,
    undefined,
    offset,
    valOffset,
    reg,
    expr,
    valExpr
};

struct UnwindRegRule
{
    UnwindRegRuleKind _M_kind = UnwindRegRuleKind::sameValue;
    intptr_t _M_off = 0;
    uint32_t _M_reg = 0;
    uint32_t _M_expr_start = 0;
    uint32_t _M_expr_end = 0;
};

enum class UnwindCfaRuleKind : unsigned char
{
    regOff,
    expr,
    undefined
};

struct UnwindCfaRule
{
    UnwindCfaRuleKind _M_kind = UnwindCfaRuleKind::undefined;
    uint32_t _M_reg = 0;
    intptr_t _M_off = 0;
    uint32_t _M_expr_start = 0;
    uint32_t _M_expr_end = 0;
};

struct UnwindCfState
{
    UnwindCfaRule _M_cfa = {};
    UnwindRegRule _M_rules[unwind_register_count] = {};
};

struct UnwindReader
{
    const uint8_t *_M_data = nullptr;
    size_t _M_len = 0;
    size_t _M_pos = 0;
    uintptr_t _M_base = 0;
    uintptr_t _M_datarel_base = 0;

    [[nodiscard]]
    constexpr size_t
    remaining () const noexcept
    {
        return this->_M_len > this->_M_pos ? this->_M_len - this->_M_pos : 0;
    }

    [[nodiscard]]
    constexpr uintptr_t
    addr () const noexcept
    {
        return this->_M_base + this->_M_pos;
    }

    bool
    byte (uint8_t &out) noexcept
    {
        if (this->_M_pos >= this->_M_len)
            return false;
        out = this->_M_data[this->_M_pos++];
        return true;
    }

    bool
    bytes (size_t n, const uint8_t *&out) noexcept
    {
        if (n > this->remaining ())
            return false;
        out = this->_M_data + this->_M_pos;
        this->_M_pos += n;
        return true;
    }

    bool
    u16 (uint16_t &out) noexcept
    {
        const uint8_t *p = nullptr;
        if (!bytes (2, p))
            return false;
        out = unwind_le16 (p);
        return true;
    }

    bool
    u32 (uint32_t &out) noexcept
    {
        const uint8_t *p = nullptr;
        if (!bytes (4, p))
            return false;
        out = unwind_le32 (p);
        return true;
    }

    bool
    u64 (uint64_t &out) noexcept
    {
        const uint8_t *p = nullptr;
        if (!bytes (8, p))
            return false;
        out = unwind_le64 (p);
        return true;
    }

    bool
    usize_width (uint8_t width, uintptr_t &out) noexcept
    {
        switch (width)
            {
            case 8:
                {
                    uint64_t v = 0;
                    if (!u64 (v))
                        return false;
                    out = static_cast<uintptr_t> (v);
                    return true;
                }
            case 4:
                {
                    uint32_t v = 0;
                    if (!u32 (v))
                        return false;
                    out = v;
                    return true;
                }
            case 2:
                {
                    uint16_t v = 0;
                    if (!u16 (v))
                        return false;
                    out = v;
                    return true;
                }
            case 1:
                {
                    uint8_t v = 0;
                    if (!byte (v))
                        return false;
                    out = v;
                    return true;
                }
            default:
                return false;
            }
    }

    bool
    uleb (uint64_t &out) noexcept
    {
        uint64_t result = 0;
        uint32_t shift = 0;
        for (;;)
            {
                uint8_t b = 0;
                if (!byte (b))
                    return false;
                result |= static_cast<uint64_t> (b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                    break;
                shift += 7;
                if (shift >= 64)
                    return false;
            }
        out = result;
        return true;
    }

    bool
    sleb (int64_t &out) noexcept
    {
        uint64_t result = 0;
        uint32_t shift = 0;
        uint8_t b = 0;
        for (;;)
            {
                if (!byte (b))
                    return false;
                result |= static_cast<uint64_t> (b & 0x7F) << shift;
                shift += 7;
                if ((b & 0x80) == 0)
                    break;
                if (shift >= 64)
                    return false;
            }
        if (shift < 64 && (b & 0x40) != 0)
            result |= (~uint64_t{0}) << shift;
        out = static_cast<int64_t> (result);
        return true;
    }

    bool
    cstr (const uint8_t *&out, size_t &out_len) noexcept
    {
        const size_t start = this->_M_pos;
        for (;;)
            {
                uint8_t b = 0;
                if (!byte (b))
                    return false;
                if (b == 0)
                    {
                        out = this->_M_data + start;
                        out_len = this->_M_pos - 1 - start;
                        return true;
                    }
            }
    }

    bool
    read_encoded_raw (uint8_t enc, uintptr_t &out) noexcept
    {
        switch (enc & 0x0F)
            {
            case 0x00:
                return usize_width (static_cast<uint8_t> (sizeof (uintptr_t)),
                                    out);
            case 0x01:
                {
                    uint64_t v = 0;
                    if (!uleb (v))
                        return false;
                    out = static_cast<uintptr_t> (v);
                    return true;
                }
            case 0x02:
                {
                    uint16_t v = 0;
                    if (!u16 (v))
                        return false;
                    out = v;
                    return true;
                }
            case 0x03:
                {
                    uint32_t v = 0;
                    if (!u32 (v))
                        return false;
                    out = v;
                    return true;
                }
            case 0x04:
                {
                    uint64_t v = 0;
                    if (!u64 (v))
                        return false;
                    out = v;
                    return true;
                }
            case 0x09:
                {
                    int64_t v = 0;
                    if (!sleb (v))
                        return false;
                    out = static_cast<uintptr_t> (v);
                    return true;
                }
            case 0x0A:
                {
                    uint16_t v = 0;
                    if (!u16 (v))
                        return false;
                    out = static_cast<uintptr_t> (static_cast<int16_t> (v));
                    return true;
                }
            case 0x0B:
                {
                    uint32_t v = 0;
                    if (!u32 (v))
                        return false;
                    out = static_cast<uintptr_t> (static_cast<int32_t> (v));
                    return true;
                }
            case 0x0C:
                {
                    uint64_t v = 0;
                    if (!u64 (v))
                        return false;
                    out = static_cast<uintptr_t> (static_cast<int64_t> (v));
                    return true;
                }
            default:
                return false;
            }
    }

    bool
    read_encoded (uint8_t enc, uintptr_t field, uintptr_t &out) noexcept
    {
        if (enc == 0xFF)
            return false;
        uintptr_t raw = 0;
        if (!read_encoded_raw (enc, raw))
            return false;
        uintptr_t value = raw;
        switch (enc & 0x70)
            {
            case 0x00:
            case 0x20:
            case 0x40:
            case 0x50:
                value = raw;
                break;
            case 0x10:
                value = raw + field;
                break;
            case 0x30:
                value = raw + this->_M_datarel_base;
                break;
            default:
                return false;
            }
        if ((enc & 0x80) != 0)
            {
                uintptr_t indirect = 0;
                if (!unwind_read_word (value, indirect))
                    return false;
                value = indirect;
            }
        out = value;
        return true;
    }
};

static UnwindReader
unwind_reader (const uint8_t *data, size_t len, uintptr_t base,
               uintptr_t datarel) noexcept
{
    UnwindReader reader;
    reader._M_data = data;
    reader._M_len = len;
    reader._M_pos = 0;
    reader._M_base = base;
    reader._M_datarel_base = datarel;
    return reader;
}

struct UnwindCie
{
    uint64_t _M_code_factor = 1;
    intptr_t _M_data_factor = 0;
    uint32_t _M_ret_reg = 0;
    uint8_t _M_fde_enc = 0x00;
    uint8_t _M_lsda_enc = 0xFF;
    bool _M_is_signal = false;
    uint8_t _M_address_size = static_cast<uint8_t> (sizeof (uintptr_t));
    uint8_t _M_version = 0;
    size_t _M_insts_off = 0;
    size_t _M_insts_end = 0;
};

struct UnwindFde
{
    UnwindCie _M_cie = {};
    uintptr_t _M_initial_location = 0;
    uintptr_t _M_address_range = 0;
    UnwindReader _M_insts = {};

    [[nodiscard]]
    bool
    contains (uintptr_t ip) const noexcept
    {
        if (this->_M_address_range == 0)
            return false;
        const uintptr_t start = this->_M_initial_location;
        const uintptr_t end = start + this->_M_address_range;
        if (end < start)
            return ip >= start;
        return ip >= start && ip < end;
    }
};

struct UnwindTables
{
    uintptr_t _M_eh_frame = 0;
    size_t _M_eh_frame_len = 0;
    bool _M_has_hdr = false;
    uintptr_t _M_hdr = 0;
    size_t _M_hdr_len = 0;
    uintptr_t _M_datarel_base = 0;
};

enum class UnwindStep : unsigned char
{
    advanced,
    stop,
    unavailable
};

static bool
unwind_parse_cie (UnwindReader &reader, size_t entry_end,
                  UnwindCie &out) noexcept
{
    uint8_t version = 0;
    if (!reader.byte (version))
        return false;
    if (version != 1 && version != 3 && version != 4)
        return false;
    const uint8_t *aug = nullptr;
    size_t aug_len = 0;
    if (!reader.cstr (aug, aug_len))
        return false;
    uint8_t address_size = uint8_t (sizeof (uintptr_t));
    if (version >= 4)
        {
            uint8_t segment_size = 0;
            if (!reader.byte (address_size) || !reader.byte (segment_size))
                return false;
            (void) segment_size;
        }
    uint64_t code_factor = 0;
    int64_t data_factor = 0;
    uint64_t ret_reg = 0;
    if (!reader.uleb (code_factor) || !reader.sleb (data_factor)
        || !reader.uleb (ret_reg))
        return false;

    uint8_t fde_enc = 0x00;
    uint8_t lsda_enc = 0xFF;
    bool is_signal = false;
    if (aug_len > 0 && aug[0] == 'z')
        {
            uint64_t aug_bytes = 0;
            if (!reader.uleb (aug_bytes))
                return false;
            const size_t aug_start = reader._M_pos;
            if (aug_bytes > (entry_end > aug_start ? entry_end - aug_start : 0))
                return false;
            const size_t aug_end = aug_start + static_cast<size_t> (aug_bytes);
            for (size_t i = 1; i < aug_len; ++i)
                {
                    const char c = static_cast<char> (aug[i]);
                    if (c == 'S')
                        {
                            is_signal = true;
                        }
                    else if (c == 'R')
                        {
                            if (!reader.byte (fde_enc))
                                return false;
                        }
                    else if (c == 'L')
                        {
                            if (!reader.byte (lsda_enc))
                                return false;
                        }
                    else if (c == 'P')
                        {
                            uint8_t personality_enc = 0;
                            uintptr_t field = 0;
                            uintptr_t sink = 0;
                            if (!reader.byte (personality_enc))
                                return false;
                            field = reader.addr ();
                            if (!reader.read_encoded (personality_enc, field,
                                                      sink))
                                return false;
                        }
                    if (reader._M_pos > aug_end)
                        return false;
                }
            reader._M_pos = aug_end;
        }
    else
        {
            for (size_t i = 0; i < aug_len; ++i)
                {
                    const char c = char (aug[i]);
                    if (c == 'S')
                        {
                            is_signal = true;
                        }
                    else if (c == 'R')
                        {
                            if (!reader.byte (fde_enc))
                                return false;
                        }
                    else if (c == 'L')
                        {
                            if (!reader.byte (lsda_enc))
                                return false;
                        }
                    else if (c == 'P')
                        {
                            uint8_t personality_enc = 0;
                            uintptr_t sink = 0;
                            if (!reader.byte (personality_enc))
                                return false;
                            const uintptr_t field = reader.addr ();
                            if (!reader.read_encoded (personality_enc, field,
                                                      sink))
                                return false;
                        }
                }
        }
    out._M_code_factor = code_factor;
    out._M_data_factor = static_cast<intptr_t> (data_factor);
    out._M_ret_reg = static_cast<uint32_t> (ret_reg);
    out._M_fde_enc = fde_enc;
    out._M_lsda_enc = lsda_enc;
    out._M_is_signal = is_signal;
    out._M_address_size = address_size;
    out._M_version = version;
    out._M_insts_off = reader._M_pos;
    out._M_insts_end = entry_end < reader._M_len ? entry_end : reader._M_len;
    if (out._M_insts_off > out._M_insts_end)
        return false;
    return true;
}

static bool
unwind_parse_fde (UnwindReader &reader, size_t entry_end, const UnwindCie &cie,
                  UnwindFde &out) noexcept
{
    uintptr_t loc_field = reader.addr ();
    uintptr_t initial_location = 0;
    if (!reader.read_encoded (cie._M_fde_enc, loc_field, initial_location))
        return false;
    uintptr_t address_range = 0;
    if (!reader.read_encoded_raw (cie._M_fde_enc, address_range))
        return false;

    if (cie._M_lsda_enc != 0xFF)
        {
            uint64_t aug_bytes = 0;
            if (!reader.uleb (aug_bytes))
                return false;
            const size_t aug_start = reader._M_pos;
            if (aug_bytes > (entry_end > aug_start ? entry_end - aug_start : 0))
                return false;
            const size_t aug_end = aug_start + static_cast<size_t> (aug_bytes);
            const uintptr_t field = reader.addr ();
            uintptr_t sink = 0;
            if (!reader.read_encoded (cie._M_lsda_enc, field, sink))
                return false;
            reader._M_pos = aug_end;
        }
    else
        {
            const size_t save = reader._M_pos;
            uint64_t aug_bytes = 0;
            if (reader.uleb (aug_bytes))
                {
                    const size_t aug_end =
                        save + static_cast<size_t> (aug_bytes);
                    const size_t avail =
                        entry_end > save ? entry_end - save : 0;
                    if (size_t (aug_bytes) <= avail && aug_end <= entry_end)
                        reader._M_pos = aug_end;
                    else
                        reader._M_pos = save;
                }
            else
                {
                    reader._M_pos = save;
                }
        }

    const size_t insts_off = reader._M_pos;
    const size_t insts_end =
        entry_end < reader._M_len ? entry_end : reader._M_len;
    if (insts_off > insts_end)
        return false;
    out._M_cie = cie;
    out._M_initial_location = initial_location;
    out._M_address_range = address_range;
    out._M_insts._M_data = reader._M_data + insts_off;
    out._M_insts._M_len = insts_end - insts_off;
    out._M_insts._M_pos = 0;
    out._M_insts._M_base = reader._M_base + insts_off;
    out._M_insts._M_datarel_base = reader._M_datarel_base;
    return true;
}

static bool
unwind_entry_offsets (const uint8_t *data, const size_t len,
                      const size_t entry_start, size_t &body, size_t &end,
                      bool &is_cie) noexcept
{
    if (entry_start + 4 > len)
        return false;
    UnwindReader reader = unwind_reader (data, len, 0, 0);
    reader._M_pos = entry_start;
    uint32_t first = 0;
    if (!reader.u32 (first))
        return false;
    if (first == 0xFFFFFFFFu)
        {
            uint64_t wide = 0;
            if (!reader.u64 (wide))
                return false;
            const size_t entry_len = static_cast<size_t> (wide);
            if (entry_len > unwind_max_entry_len || entry_len == 0)
                return false;
            body = reader._M_pos;
            if (entry_len > len - body)
                return false;
            end = body + entry_len;
            uint64_t id = 0;
            if (!reader.u64 (id))
                return false;
            is_cie = id == 0;
            return true;
        }
    if (first == 0 || first > unwind_max_entry_len)
        return false;
    body = entry_start + 4;
    if (static_cast<size_t> (first) > len - body)
        return false;
    end = body + static_cast<size_t> (first);
    reader._M_pos = body;
    uint32_t id = 0;
    if (!reader.u32 (id))
        return false;
    is_cie = id == 0;
    return true;
}

static bool
unwind_is_wide_cie (const uint8_t *data, const size_t len,
                    const size_t cie_start) noexcept
{
    if (cie_start + 4 > len)
        return false;
    return data[cie_start] == 0xFF && data[cie_start + 1] == 0xFF
           && data[cie_start + 2] == 0xFF && data[cie_start + 3] == 0xFF;
}

static bool
unwind_scan_eh_frame (const uint8_t *data, const size_t len,
                      const uintptr_t base, const uintptr_t datarel_base,
                      const uintptr_t ip, UnwindFde &out) noexcept
{
    size_t off = 0;
    while (off + 4 <= len)
        {
            if (data[off] == 0 && data[off + 1] == 0 && data[off + 2] == 0
                && data[off + 3] == 0)
                break;
            size_t body = 0;
            size_t end = 0;
            bool is_cie = false;
            if (!unwind_entry_offsets (data, len, off, body, end, is_cie))
                break;
            if (end <= off)
                break;
            if (!is_cie)
                {
                    UnwindReader idr =
                        unwind_reader (data, len, base, datarel_base);
                    idr._M_pos = body;
                    uint32_t cie_ptr = 0;
                    if (!idr.u32 (cie_ptr))
                        {
                            break;
                        }
                    if (cie_ptr > body)
                        {
                            off = end;
                            continue;
                        }
                    const size_t cie_start = body - cie_ptr;
                    size_t cie_body = 0;
                    size_t cie_end = 0;
                    if (bool cie_is_cie = false;
                        unwind_entry_offsets (data, len, cie_start, cie_body,
                                              cie_end, cie_is_cie)
                        && cie_is_cie)
                        {
                            UnwindReader cr =
                                unwind_reader (data, len, base, datarel_base);
                            cr._M_pos =
                                cie_start
                                + (unwind_is_wide_cie (data, len, cie_start)
                                       ? 20
                                       : 8);
                            if (UnwindCie cie;
                                unwind_parse_cie (cr, cie_end, cie))
                                {
                                    UnwindReader fr = unwind_reader (
                                        data, len, base, datarel_base);
                                    fr._M_pos = body + 4;
                                    UnwindFde fde;
                                    if (unwind_parse_fde (fr, end, cie, fde)
                                        && fde.contains (ip))
                                        {
                                            out = fde;
                                            return true;
                                        }
                                }
                        }
                }
            off = end;
        }
    return false;
}

static bool
unwind_fixed_encoded_size (uint8_t enc, size_t &out) noexcept
{
    switch (enc & 0x0F)
        {
        case 0x02:
        case 0x0A:
            out = 2;
            return true;
        case 0x03:
        case 0x0B:
            out = 4;
            return true;
        case 0x04:
        case 0x0C:
            out = 8;
            return true;
        case 0x00:
            out = sizeof (uintptr_t);
            return true;
        default:
            return false;
        }
}

static bool
unwind_hdr_find_fde (const UnwindTables &tables, const uint8_t *frame,
                     size_t frame_len, uintptr_t ip, UnwindFde &out) noexcept
{
    if (!tables._M_has_hdr || tables._M_hdr_len < 4 || tables._M_hdr < 4096)
        return false;
    const auto *hdr = reinterpret_cast<const uint8_t *> (tables._M_hdr);
    if (hdr[0] != 1)
        return false;
    const uint8_t eh_frame_ptr_enc = hdr[1];
    const uint8_t fde_count_enc = hdr[2];
    const uint8_t table_enc = hdr[3];
    UnwindReader reader = unwind_reader (hdr + 4, tables._M_hdr_len - 4,
                                         tables._M_hdr + 4, tables._M_hdr);
    uintptr_t sink = 0;
    uintptr_t field = reader.addr ();
    if (!reader.read_encoded (eh_frame_ptr_enc, field, sink))
        return false;
    field = reader.addr ();
    uintptr_t count = 0;
    if (!reader.read_encoded (fde_count_enc, field, count))
        return false;
    if (count > (uintptr_t{1} << 24))
        count = uintptr_t{1} << 24;
    size_t entry_size = 0;
    if (!unwind_fixed_encoded_size (table_enc, entry_size))
        return false;
    if (entry_size != 0 && size_t (count) > (SIZE_MAX / 2 / entry_size))
        return false;
    const size_t table_bytes = size_t (count) * entry_size * 2;
    if (reader.remaining () < table_bytes)
        return false;
    const uint8_t *table = hdr + 4 + reader._M_pos;
    const uintptr_t table_addr = tables._M_hdr + 4 + reader._M_pos;

    size_t lo = 0;
    size_t hi = size_t (count);
    size_t best = 0;
    bool found_best = false;
    while (lo < hi)
        {
            const size_t mid = lo + (hi - lo) / 2;
            UnwindReader er =
                unwind_reader (table, table_bytes, table_addr, tables._M_hdr);
            er._M_pos = mid * entry_size * 2;
            const uintptr_t loc_field = er.addr ();
            uintptr_t loc = 0;
            if (!er.read_encoded (table_enc, loc_field, loc))
                return false;
            if (loc <= ip)
                {
                    best = mid;
                    found_best = true;
                    lo = mid + 1;
                }
            else
                {
                    hi = mid;
                }
        }
    if (!found_best)
        return false;

    UnwindReader er =
        unwind_reader (table, table_bytes, table_addr, tables._M_hdr);
    er._M_pos = best * entry_size * 2;
    uintptr_t loc_field = er.addr ();
    uintptr_t sink_loc = 0;
    if (!er.read_encoded (table_enc, loc_field, sink_loc))
        return false;
    loc_field = er.addr ();
    uintptr_t fde_addr = 0;
    if (!er.read_encoded (table_enc, loc_field, fde_addr))
        return false;

    if (fde_addr < tables._M_eh_frame)
        return false;
    const size_t off = size_t (fde_addr - tables._M_eh_frame);
    if (off >= frame_len)
        return false;
    size_t body = 0;
    size_t end = 0;
    bool is_cie = false;
    if (!unwind_entry_offsets (frame, frame_len, off, body, end, is_cie)
        || is_cie)
        return false;
    UnwindReader idr = unwind_reader (frame, frame_len, tables._M_eh_frame,
                                      tables._M_datarel_base);
    idr._M_pos = body;
    uint32_t cie_ptr = 0;
    if (!idr.u32 (cie_ptr))
        return false;
    if (cie_ptr > body)
        return false;
    const size_t cie_start = body - cie_ptr;
    size_t cie_body = 0;
    size_t cie_end = 0;
    bool cie_is_cie = false;
    if (!unwind_entry_offsets (frame, frame_len, cie_start, cie_body, cie_end,
                               cie_is_cie)
        || !cie_is_cie)
        return false;
    UnwindReader cr = unwind_reader (frame, frame_len, tables._M_eh_frame,
                                     tables._M_datarel_base);
    cr._M_pos =
        cie_start + (unwind_is_wide_cie (frame, frame_len, cie_start) ? 20 : 8);
    UnwindCie cie;
    if (!unwind_parse_cie (cr, cie_end, cie))
        return false;
    UnwindReader fr = unwind_reader (frame, frame_len, tables._M_eh_frame,
                                     tables._M_datarel_base);
    fr._M_pos = body + 4;
    UnwindFde fde;
    if (!unwind_parse_fde (fr, end, cie, fde))
        return false;
    if (!fde.contains (ip))
        return false;
    out = fde;
    return true;
}

static bool
unwind_find_fde (const UnwindTables &tables, const uint8_t *frame,
                 size_t frame_len, uintptr_t ip, UnwindFde &out) noexcept
{
    if (unwind_hdr_find_fde (tables, frame, frame_len, ip, out))
        return true;
    return unwind_scan_eh_frame (frame, frame_len, tables._M_eh_frame,
                                 tables._M_datarel_base, ip, out);
}

static bool
unwind_eval_expr (const UnwindReader &in, const UnwindRegs &regs, uintptr_t cfa,
                  uint8_t address_size, uintptr_t &out) noexcept
{
    UnwindReader reader = in;
    uintptr_t stack[unwind_expr_stack_limit] = {};
    size_t sp = 0;
    const auto push = [&] (uintptr_t value) noexcept -> bool
        {
            if (sp >= unwind_expr_stack_limit)
                return false;
            stack[sp++] = value;
            return true;
        };
    const auto peek = [&] (size_t depth, uintptr_t &value) noexcept -> bool
        {
            if (depth >= sp)
                return false;
            value = stack[sp - 1 - depth];
            return true;
        };
    const auto binary_op = [&] (auto op) noexcept -> bool
        {
            if (sp < 2)
                return false;
            stack[sp - 2] = op (stack[sp - 2], stack[sp - 1]);
            --sp;
            return true;
        };

    while (reader.remaining () > 0)
        {
            uint8_t op = 0;
            if (!reader.byte (op))
                return false;
            switch (op)
                {
                case 0x03:
                    {
                        uintptr_t v = 0;
                        if (!reader.usize_width (address_size, v) || !push (v))
                            return false;
                        break;
                    }
                case 0x06:
                    {
                        uintptr_t addr = 0;
                        uintptr_t value = 0;
                        if (!peek (0, addr) || !unwind_read_word (addr, value))
                            return false;
                        stack[sp - 1] = value;
                        break;
                    }
                case 0x08:
                    {
                        uint8_t v = 0;
                        if (!reader.byte (v) || !push (v))
                            return false;
                        break;
                    }
                case 0x09:
                    {
                        uint8_t v = 0;
                        if (!reader.byte (v) || !push (uintptr_t (int8_t (v))))
                            return false;
                        break;
                    }
                case 0x0A:
                    {
                        uint16_t v = 0;
                        if (!reader.u16 (v) || !push (v))
                            return false;
                        break;
                    }
                case 0x0B:
                    {
                        uint16_t v = 0;
                        if (!reader.u16 (v) || !push (uintptr_t (int16_t (v))))
                            return false;
                        break;
                    }
                case 0x0C:
                    {
                        uint32_t v = 0;
                        if (!reader.u32 (v) || !push (v))
                            return false;
                        break;
                    }
                case 0x0D:
                    {
                        uint32_t v = 0;
                        if (!reader.u32 (v) || !push (uintptr_t (int32_t (v))))
                            return false;
                        break;
                    }
                case 0x0E:
                    {
                        uint64_t v = 0;
                        if (!reader.u64 (v) || !push (uintptr_t (v)))
                            return false;
                        break;
                    }
                case 0x0F:
                    {
                        uint64_t v = 0;
                        if (!reader.u64 (v) || !push (uintptr_t (int64_t (v))))
                            return false;
                        break;
                    }
                case 0x10:
                    {
                        uint64_t v = 0;
                        if (!reader.uleb (v) || !push (uintptr_t (v)))
                            return false;
                        break;
                    }
                case 0x11:
                    {
                        int64_t v = 0;
                        if (!reader.sleb (v) || !push (uintptr_t (v)))
                            return false;
                        break;
                    }
                case 0x12:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v) || !push (v))
                            return false;
                        break;
                    }
                case 0x13:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v))
                            return false;
                        --sp;
                        break;
                    }
                case 0x14:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v) || !push (v))
                            return false;
                        break;
                    }
                case 0x15:
                    {
                        uint8_t depth = 0;
                        uintptr_t v = 0;
                        if (!reader.byte (depth))
                            return false;
                        if (size_t (depth) + 1 > sp)
                            return false;
                        v = stack[sp - 1 - size_t (depth)];
                        if (!push (v))
                            return false;
                        break;
                    }
                case 0x16:
                    {
                        if (sp < 2)
                            return false;
                        const uintptr_t tmp = stack[sp - 1];
                        stack[sp - 1] = stack[sp - 2];
                        stack[sp - 2] = tmp;
                        break;
                    }
                case 0x19:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v))
                            return false;
                        stack[sp - 1] =
                            uintptr_t (intptr_t (int64_t (v)) < 0
                                           ? (0 - uint64_t (int64_t (v)))
                                           : uint64_t (int64_t (v)));
                        break;
                    }
                case 0x1A:
                    {
                        if (sp < 2)
                            return false;
                        stack[sp - 2] &= stack[sp - 1];
                        --sp;
                        break;
                    }
                case 0x1B:
                    {
                        if (sp < 2)
                            return false;
                        const int64_t b = int64_t (stack[sp - 1]);
                        if (b == 0)
                            return false;
                        stack[sp - 2] = uintptr_t (int64_t (stack[sp - 2]) / b);
                        --sp;
                        break;
                    }
                case 0x1C:
                    {
                        if (!binary_op ([] (uintptr_t a, uintptr_t b) noexcept
                                            { return a - b; }))
                            return false;
                        break;
                    }
                case 0x1D:
                    {
                        if (sp < 2)
                            return false;
                        const int64_t b = int64_t (stack[sp - 1]);
                        if (b == 0)
                            return false;
                        stack[sp - 2] = uintptr_t (int64_t (stack[sp - 2]) % b);
                        --sp;
                        break;
                    }
                case 0x1E:
                    {
                        if (!binary_op ([] (uintptr_t a, uintptr_t b) noexcept
                                            { return a * b; }))
                            return false;
                        break;
                    }
                case 0x1F:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v))
                            return false;
                        stack[sp - 1] = uintptr_t (~v + 1);
                        break;
                    }
                case 0x20:
                    {
                        uintptr_t v = 0;
                        if (!peek (0, v))
                            return false;
                        stack[sp - 1] = ~v;
                        break;
                    }
                case 0x21:
                    {
                        if (!binary_op ([] (uintptr_t a, uintptr_t b) noexcept
                                            { return a | b; }))
                            return false;
                        break;
                    }
                case 0x22:
                    {
                        if (!binary_op ([] (uintptr_t a, uintptr_t b) noexcept
                                            { return a + b; }))
                            return false;
                        break;
                    }
                case 0x23:
                    {
                        uint64_t c = 0;
                        uintptr_t v = 0;
                        if (!reader.uleb (c) || !peek (0, v))
                            return false;
                        stack[sp - 1] = v + uintptr_t (c);
                        break;
                    }
                case 0x24:
                    {
                        if (sp < 2)
                            return false;
                        stack[sp - 2] <<= stack[sp - 1];
                        --sp;
                        break;
                    }
                case 0x25:
                    {
                        if (sp < 2)
                            return false;
                        stack[sp - 2] >>= stack[sp - 1];
                        --sp;
                        break;
                    }
                case 0x26:
                    {
                        if (sp < 2)
                            return false;
                        stack[sp - 2] = uintptr_t (intptr_t (stack[sp - 2])
                                                   >> stack[sp - 1]);
                        --sp;
                        break;
                    }
                case 0x27:
                    {
                        if (!binary_op ([] (uintptr_t a, uintptr_t b) noexcept
                                            { return a ^ b; }))
                            return false;
                        break;
                    }
                case 0x28:
                    {
                        uint16_t off = 0;
                        uintptr_t cond = 0;
                        if (!reader.u16 (off) || !peek (0, cond))
                            return false;
                        --sp;
                        if (cond != 0)
                            {
                                const size_t target =
                                    reader._M_pos + intptr_t (int16_t (off));
                                if (target > reader._M_len)
                                    return false;
                                reader._M_pos = target;
                            }
                        break;
                    }
                case 0x29:
                case 0x2A:
                case 0x2B:
                case 0x2C:
                case 0x2D:
                case 0x2E:
                    {
                        if (sp < 2)
                            return false;
                        const uintptr_t a = stack[sp - 2];
                        const uintptr_t b = stack[sp - 1];
                        bool result = false;
                        switch (op)
                            {
                            case 0x29:
                                result = a == b;
                                break;
                            case 0x2A:
                                result = int64_t (a) >= int64_t (b);
                                break;
                            case 0x2B:
                                result = int64_t (a) > int64_t (b);
                                break;
                            case 0x2C:
                                result = int64_t (a) <= int64_t (b);
                                break;
                            case 0x2D:
                                result = int64_t (a) < int64_t (b);
                                break;
                            default:
                                result = a != b;
                                break;
                            }
                        stack[sp - 2] = result ? 1u : 0u;
                        --sp;
                        break;
                    }
                case 0x2F:
                    {
                        uint16_t off = 0;
                        if (!reader.u16 (off))
                            return false;
                        const size_t target =
                            reader._M_pos + intptr_t (int16_t (off));
                        if (target > reader._M_len)
                            return false;
                        reader._M_pos = target;
                        break;
                    }
                default:
                    if (op >= 0x30 && op <= 0x4F)
                        {
                            if (!push (op - 0x30))
                                return false;
                        }
                    else if (op >= 0x50 && op <= 0x6F)
                        {
                            const size_t reg = op - 0x50;
                            if (reg >= unwind_register_count)
                                return false;
                            if (!push (regs._M_gpr[reg]))
                                return false;
                        }
                    else if (op >= 0x70 && op <= 0x8F)
                        {
                            const size_t reg = op - 0x70;
                            int64_t off = 0;
                            if (reg >= unwind_register_count
                                || !reader.sleb (off))
                                return false;
                            if (!push (uintptr_t (int64_t (regs._M_gpr[reg])
                                                  + off)))
                                return false;
                        }
                    else if (op == 0x90)
                        {
                            uint64_t reg = 0;
                            if (!reader.uleb (reg)
                                || reg >= unwind_register_count)
                                return false;
                            if (!push (regs._M_gpr[size_t (reg)]))
                                return false;
                        }
                    else if (op == 0x91)
                        {
                            int64_t off = 0;
                            if (!reader.sleb (off))
                                return false;
                            if (!push (uintptr_t (int64_t (cfa) + off)))
                                return false;
                        }
                    else if (op == 0x92)
                        {
                            uint64_t reg = 0;
                            int64_t off = 0;
                            if (!reader.uleb (reg) || !reader.sleb (off)
                                || reg >= unwind_register_count)
                                return false;
                            if (!push (uintptr_t (
                                    int64_t (regs._M_gpr[size_t (reg)]) + off)))
                                return false;
                        }
                    else if (op == 0x94)
                        {
                            uint8_t size = 0;
                            uintptr_t addr = 0;
                            if (!reader.byte (size) || !peek (0, addr))
                                return false;
                            if (size == 0)
                                {
                                    stack[sp - 1] = 0;
                                }
                            else if (addr < 4096)
                                {
                                    return false;
                                }
                            else
                                {
                                    const auto *bytes = reinterpret_cast<
                                        const unsigned char *> (addr);
                                    uintptr_t value = 0;
                                    for (size_t i = 0;
                                         i < size_t (size)
                                         && i < sizeof (uintptr_t);
                                         ++i)
                                        value |= uintptr_t{bytes[i]} << (8 * i);
                                    stack[sp - 1] = value;
                                }
                            break;
                        }
                    else if (op == 0x96)
                        {
                            break;
                        }
                    else if (op == 0x9C)
                        {
                            if (!push (cfa))
                                return false;
                        }
                    else
                        {
                            return false;
                        }
                    break;
                }
        }
    if (sp == 0)
        return false;
    out = stack[sp - 1];
    return true;
}

static bool
unwind_exec_cfa (UnwindReader &reader, UnwindCfState &state,
                 const UnwindCfState &base_state, UnwindCfState state_stack[],
                 size_t &stack_len, uint64_t code_factor, intptr_t data_factor,
                 uint8_t address_size, uintptr_t start_loc, uintptr_t target,
                 bool tracking, uintptr_t &out_loc) noexcept
{
    uintptr_t loc = start_loc;
    while (reader.remaining () > 0)
        {
            if (tracking && loc > target)
                break;
            uint8_t op = 0;
            if (!reader.byte (op))
                return false;
            const uint8_t short_hi = op & 0xC0;
            const size_t short_reg = op & 0x3F;
            switch (short_hi)
                {
                case 0x40:
                    loc += short_reg * static_cast<size_t> (code_factor);
                    break;
                case 0x80:
                    {
                        uint64_t off = 0;
                        if (short_reg >= unwind_register_count
                            || !reader.uleb (off))
                            return false;
                        state._M_rules[short_reg]._M_kind =
                            UnwindRegRuleKind::offset;
                        state._M_rules[short_reg]._M_off =
                            intptr_t (off) * data_factor;
                        break;
                    }
                case 0xC0:
                    if (short_reg >= unwind_register_count)
                        return false;
                    state._M_rules[short_reg] = base_state._M_rules[short_reg];
                    break;
                default:
                    switch (op)
                        {
                        case 0x00:
                            break;
                        case 0x01:
                            {
                                uintptr_t v = 0;
                                if (!reader.usize_width (address_size, v))
                                    return false;
                                loc = v;
                                break;
                            }
                        case 0x02:
                            {
                                uint8_t v = 0;
                                if (!reader.byte (v))
                                    return false;
                                loc += size_t (v) * size_t (code_factor);
                                break;
                            }
                        case 0x03:
                            {
                                uint16_t v = 0;
                                if (!reader.u16 (v))
                                    return false;
                                loc += size_t (v) * size_t (code_factor);
                                break;
                            }
                        case 0x04:
                            {
                                uint32_t v = 0;
                                if (!reader.u32 (v))
                                    return false;
                                loc += size_t (v) * size_t (code_factor);
                                break;
                            }
                        case 0x05:
                            {
                                uint64_t reg = 0;
                                uint64_t off = 0;
                                if (!reader.uleb (reg) || !reader.uleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::offset;
                                state._M_rules[size_t (reg)]._M_off =
                                    intptr_t (off) * data_factor;
                                break;
                            }
                        case 0x06:
                            {
                                uint64_t reg = 0;
                                if (!reader.uleb (reg)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)] =
                                    base_state._M_rules[size_t (reg)];
                                break;
                            }
                        case 0x07:
                            {
                                uint64_t reg = 0;
                                if (!reader.uleb (reg)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::undefined;
                                break;
                            }
                        case 0x08:
                            {
                                uint64_t reg = 0;
                                if (!reader.uleb (reg)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::sameValue;
                                break;
                            }
                        case 0x09:
                            {
                                uint64_t reg = 0;
                                uint64_t src = 0;
                                if (!reader.uleb (reg) || !reader.uleb (src)
                                    || reg >= unwind_register_count
                                    || src >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::reg;
                                state._M_rules[size_t (reg)]._M_reg =
                                    uint32_t (src);
                                break;
                            }
                        case 0x0A:
                            if (stack_len >= unwind_state_stack_limit)
                                return false;
                            state_stack[stack_len++] = state;
                            break;
                        case 0x0B:
                            if (stack_len == 0)
                                return false;
                            --stack_len;
                            state = state_stack[stack_len];
                            break;
                        case 0x0C:
                            {
                                uint64_t reg = 0;
                                uint64_t off = 0;
                                if (!reader.uleb (reg) || !reader.uleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_cfa._M_kind =
                                    UnwindCfaRuleKind::regOff;
                                state._M_cfa._M_reg = uint32_t (reg);
                                state._M_cfa._M_off = intptr_t (off);
                                break;
                            }
                        case 0x0D:
                            {
                                uint64_t reg = 0;
                                if (!reader.uleb (reg)
                                    || reg >= unwind_register_count)
                                    return false;
                                const intptr_t off =
                                    state._M_cfa._M_kind
                                            == UnwindCfaRuleKind::regOff
                                        ? state._M_cfa._M_off
                                        : 0;
                                state._M_cfa._M_kind =
                                    UnwindCfaRuleKind::regOff;
                                state._M_cfa._M_reg = uint32_t (reg);
                                state._M_cfa._M_off = off;
                                break;
                            }
                        case 0x0E:
                            {
                                uint64_t off = 0;
                                if (!reader.uleb (off))
                                    return false;
                                if (state._M_cfa._M_kind
                                    != UnwindCfaRuleKind::regOff)
                                    return false;
                                state._M_cfa._M_off = intptr_t (off);
                                break;
                            }
                        case 0x0F:
                            {
                                uint64_t len = 0;
                                if (!reader.uleb (len))
                                    return false;
                                const size_t start = reader._M_pos;
                                if (size_t (len) > reader._M_len - start)
                                    return false;
                                const size_t end = start + size_t (len);
                                state._M_cfa._M_kind = UnwindCfaRuleKind::expr;
                                state._M_cfa._M_expr_start = uint32_t (start);
                                state._M_cfa._M_expr_end = uint32_t (end);
                                reader._M_pos = end;
                                break;
                            }
                        case 0x10:
                            {
                                uint64_t reg = 0;
                                uint64_t len = 0;
                                if (!reader.uleb (reg) || !reader.uleb (len)
                                    || reg >= unwind_register_count)
                                    return false;
                                const size_t start = reader._M_pos;
                                if (size_t (len) > reader._M_len - start)
                                    return false;
                                const size_t end = start + size_t (len);
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::expr;
                                state._M_rules[size_t (reg)]._M_expr_start =
                                    uint32_t (start);
                                state._M_rules[size_t (reg)]._M_expr_end =
                                    uint32_t (end);
                                reader._M_pos = end;
                                break;
                            }
                        case 0x11:
                            {
                                uint64_t reg = 0;
                                int64_t off = 0;
                                if (!reader.uleb (reg) || !reader.sleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::offset;
                                state._M_rules[size_t (reg)]._M_off =
                                    off * data_factor;
                                break;
                            }
                        case 0x12:
                            {
                                uint64_t reg = 0;
                                int64_t off = 0;
                                if (!reader.uleb (reg) || !reader.sleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_cfa._M_kind =
                                    UnwindCfaRuleKind::regOff;
                                state._M_cfa._M_reg = uint32_t (reg);
                                state._M_cfa._M_off = off * data_factor;
                                break;
                            }
                        case 0x13:
                            {
                                int64_t off = 0;
                                if (!reader.sleb (off))
                                    return false;
                                if (state._M_cfa._M_kind
                                    != UnwindCfaRuleKind::regOff)
                                    return false;
                                state._M_cfa._M_off = off * data_factor;
                                break;
                            }
                        case 0x14:
                            {
                                uint64_t reg = 0;
                                uint64_t off = 0;
                                if (!reader.uleb (reg) || !reader.uleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::valOffset;
                                state._M_rules[size_t (reg)]._M_off =
                                    intptr_t (off) * data_factor;
                                break;
                            }
                        case 0x15:
                            {
                                uint64_t reg = 0;
                                int64_t off = 0;
                                if (!reader.uleb (reg) || !reader.sleb (off)
                                    || reg >= unwind_register_count)
                                    return false;
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::valOffset;
                                state._M_rules[size_t (reg)]._M_off =
                                    off * data_factor;
                                break;
                            }
                        case 0x16:
                            {
                                uint64_t reg = 0;
                                uint64_t len = 0;
                                if (!reader.uleb (reg) || !reader.uleb (len)
                                    || reg >= unwind_register_count)
                                    return false;
                                const size_t start = reader._M_pos;
                                if (size_t (len) > reader._M_len - start)
                                    return false;
                                const size_t end = start + size_t (len);
                                state._M_rules[size_t (reg)]._M_kind =
                                    UnwindRegRuleKind::valExpr;
                                state._M_rules[size_t (reg)]._M_expr_start =
                                    uint32_t (start);
                                state._M_rules[size_t (reg)]._M_expr_end =
                                    uint32_t (end);
                                reader._M_pos = end;
                                break;
                            }
                        case 0x2D:
                            break;
                        case 0x2E:
                            {
                                uint64_t args_size = 0;
                                if (!reader.uleb (args_size))
                                    return false;
                                break;
                            }
                        default:
                            return false;
                        }
                    break;
                }
        }
    out_loc = loc;
    return true;
}

static bool
unwind_apply_rules (const UnwindCfState &state, const UnwindRegs &regs,
                    const UnwindCie &cie, const uint8_t *frame_data,
                    size_t frame_len, uintptr_t frame_base,
                    UnwindState &out) noexcept
{
    uintptr_t cfa = 0;
    switch (state._M_cfa._M_kind)
        {
        case UnwindCfaRuleKind::regOff:
            {
                if (state._M_cfa._M_reg >= unwind_register_count)
                    return false;
                const uintptr_t value = regs._M_gpr[state._M_cfa._M_reg];
                cfa = uintptr_t (intptr_t (value) + state._M_cfa._M_off);
                break;
            }
        case UnwindCfaRuleKind::expr:
            {
                const size_t end = size_t (state._M_cfa._M_expr_end) < frame_len
                                       ? size_t (state._M_cfa._M_expr_end)
                                       : frame_len;
                const size_t start = state._M_cfa._M_expr_start;
                if (start > end)
                    return false;
                UnwindReader expr = unwind_reader (
                    frame_data + start, end - start, frame_base + start, 0);
                if (!unwind_eval_expr (expr, regs, 0, cie._M_address_size, cfa))
                    return false;
                break;
            }
        default:
            return false;
        }

    if (cfa < 4096)
        return false;

    UnwindRegs next = regs;
    for (size_t i = 0; i < unwind_register_count; ++i)
        {
            const UnwindRegRule &rule = state._M_rules[i];
            uintptr_t value = 0;
            switch (rule._M_kind)
                {
                case UnwindRegRuleKind::sameValue:
                    value = regs._M_gpr[i];
                    break;
                case UnwindRegRuleKind::undefined:
                    value = 0;
                    break;
                case UnwindRegRuleKind::offset:
                    {
                        const uintptr_t addr =
                            uintptr_t (intptr_t (cfa) + rule._M_off);
                        if (!unwind_read_word (addr, value))
                            return false;
                        break;
                    }
                case UnwindRegRuleKind::valOffset:
                    value = uintptr_t (intptr_t (cfa) + rule._M_off);
                    break;
                case UnwindRegRuleKind::reg:
                    if (rule._M_reg >= unwind_register_count)
                        return false;
                    value = regs._M_gpr[rule._M_reg];
                    break;
                case UnwindRegRuleKind::expr:
                    {
                        const size_t end = size_t (rule._M_expr_end) < frame_len
                                               ? size_t (rule._M_expr_end)
                                               : frame_len;
                        const size_t start = rule._M_expr_start;
                        if (start > end)
                            return false;
                        UnwindReader expr =
                            unwind_reader (frame_data + start, end - start,
                                           frame_base + start, 0);
                        uintptr_t addr = 0;
                        if (!unwind_eval_expr (expr, regs, cfa,
                                               cie._M_address_size, addr))
                            return false;
                        if (!unwind_read_word (addr, value))
                            return false;
                        break;
                    }
                case UnwindRegRuleKind::valExpr:
                    {
                        const size_t end = size_t (rule._M_expr_end) < frame_len
                                               ? size_t (rule._M_expr_end)
                                               : frame_len;
                        const size_t start = rule._M_expr_start;
                        if (start > end)
                            return false;
                        UnwindReader expr =
                            unwind_reader (frame_data + start, end - start,
                                           frame_base + start, 0);
                        if (!unwind_eval_expr (expr, regs, cfa,
                                               cie._M_address_size, value))
                            return false;
                        break;
                    }
                }
            next._M_gpr[i] = value;
        }

    next._M_gpr[unwind_sp_index ()] = cfa;
    if (cie._M_ret_reg >= unwind_register_count)
        return false;
    const uintptr_t next_ip = next._M_gpr[cie._M_ret_reg];
    if (next_ip == 0)
        return false;
    out = UnwindState{};
    out._M_regs = next;
    out._M_regs._M_ip = next_ip;
    out._M_sp = cfa;
    return true;
}

static UnwindStep
unwind_cfi_step (const UnwindState &state, const UnwindTables &tables,
                 const uint8_t *frame, const size_t frame_len,
                 UnwindState &out) noexcept
{
    const uintptr_t ip = state._M_regs._M_ip;
    UnwindFde fde;
    if (!unwind_find_fde (tables, frame, frame_len, ip, fde))
        return UnwindStep::unavailable;

    UnwindCfState cf;
    UnwindCfState base;
    UnwindCfState stack_buf[unwind_state_stack_limit];
    size_t stack_len = 0;

    const size_t cie_start = fde._M_cie._M_insts_off < frame_len
                                 ? fde._M_cie._M_insts_off
                                 : frame_len;
    const size_t cie_end = fde._M_cie._M_insts_end < frame_len
                               ? fde._M_cie._M_insts_end
                               : frame_len;
    if (cie_start > cie_end)
        return UnwindStep::unavailable;
    UnwindReader cie_reader =
        unwind_reader (frame + cie_start, cie_end - cie_start,
                       tables._M_eh_frame + cie_start, tables._M_datarel_base);
    uintptr_t loc = 0;
    if (!unwind_exec_cfa (cie_reader, cf, base, stack_buf, stack_len,
                          fde._M_cie._M_code_factor, fde._M_cie._M_data_factor,
                          fde._M_cie._M_address_size, 0, 0, false, loc))
        return UnwindStep::unavailable;
    base = cf;
    stack_len = 0;

    UnwindReader fde_reader = fde._M_insts;
    if (!unwind_exec_cfa (fde_reader, cf, base, stack_buf, stack_len,
                          fde._M_cie._M_code_factor, fde._M_cie._M_data_factor,
                          fde._M_cie._M_address_size, fde._M_initial_location,
                          ip, true, loc))
        return UnwindStep::unavailable;

    UnwindState next;
    if (!unwind_apply_rules (cf, state._M_regs, fde._M_cie, frame, frame_len,
                             tables._M_eh_frame, next))
        return UnwindStep::stop;
    if (fde._M_cie._M_is_signal)
        next._M_stop = true;
    out = next;
    return UnwindStep::advanced;
}
#if defined(ZZS_UNWIND_ELF)
inline constexpr uint32_t unwind_pt_load = 1;
inline constexpr uint32_t unwind_pt_gnu_eh_frame = 0x6474E550;
inline constexpr uint32_t unwind_pf_x = 1;

struct UnwindFound
{
    bool _M_has_base = false;
    uintptr_t _M_base = 0;
    bool _M_has_eh_frame = false;
    uintptr_t _M_eh_frame = 0;
    uintptr_t _M_eh_frame_len = 0;
    bool _M_has_hdr = false;
    uintptr_t _M_hdr = 0;
    uintptr_t _M_hdr_len = 0;
    uintptr_t _M_datarel_base = 0;
};

struct UnwindSearch
{
    uintptr_t _M_ip = 0;
    UnwindFound found = {};
};

static void
unwind_load_range (const ::dl_phdr_info *info, const ElfW (Phdr) & ph,
                   uintptr_t &start, uintptr_t &end) noexcept
{
    start = uintptr_t (info->dlpi_addr) + uintptr_t (ph.p_vaddr);
    end = start + uintptr_t (ph.p_memsz);
}

static bool
unwind_decode_at (const uint8_t *data, size_t len, uint8_t enc, uintptr_t field,
                  uintptr_t datarel, uintptr_t &out) noexcept
{
    if (enc == 0xFF || len == 0)
        return false;
    uintptr_t raw = 0;
    switch (enc & 0x0F)
        {
        case 0x00:
            {
                const size_t n = sizeof (uintptr_t);
                if (len < n)
                    return false;
                raw = 0;
                for (size_t i = 0; i < n; ++i)
                    raw |= uintptr_t{data[i]} << (8 * i);
                break;
            }
        case 0x01:
            {
                uint64_t v = 0;
                uint32_t shift = 0;
                size_t pos = 0;
                for (;;)
                    {
                        if (pos >= len)
                            return false;
                        const uint8_t b = data[pos++];
                        v |= uint64_t (b & 0x7F) << shift;
                        if ((b & 0x80) == 0)
                            break;
                        shift += 7;
                        if (shift >= 64)
                            return false;
                    }
                raw = uintptr_t (v);
                break;
            }
        case 0x02:
            if (len < 2)
                return false;
            raw = unwind_le16 (data);
            break;
        case 0x03:
            if (len < 4)
                return false;
            raw = unwind_le32 (data);
            break;
        case 0x04:
            if (len < 8)
                return false;
            raw = uintptr_t (unwind_le64 (data));
            break;
        case 0x0B:
            if (len < 4)
                return false;
            raw = uintptr_t (int32_t (unwind_le32 (data)));
            break;
        case 0x0C:
            if (len < 8)
                return false;
            raw = uintptr_t (int64_t (unwind_le64 (data)));
            break;
        default:
            return false;
        }
    switch (enc & 0x70)
        {
        case 0x10:
            out = raw + field;
            break;
        case 0x30:
            out = raw + datarel;
            break;
        default:
            out = raw;
            break;
        }
    return true;
}

static bool
unwind_hdr_eh_frame (const uint8_t *hdr, size_t hdr_len, uintptr_t hdr_addr,
                     const ::dl_phdr_info *info, uintptr_t &out_eh_frame,
                     uintptr_t &out_eh_frame_len) noexcept
{
    if (hdr_addr < 4096 || hdr_len < 4)
        return false;
    if (hdr[0] != 1)
        return false;
    const uint8_t enc = hdr[1];
    uintptr_t value = 0;
    if (!unwind_decode_at (hdr + 4, hdr_len - 4, enc, hdr_addr + 4, 0, value))
        return false;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type != unwind_pt_load)
                continue;
            uintptr_t start = 0;
            uintptr_t end = 0;
            unwind_load_range (info, ph, start, end);
            if (value >= start && value < end)
                {
                    out_eh_frame = value;
                    out_eh_frame_len = end - value;
                    return true;
                }
        }
    return false;
}

static bool
unwind_in_load_range (const ::dl_phdr_info *info, uintptr_t addr,
                      size_t len) noexcept
{
    if (len == 0 || addr < 4096)
        return false;
    if (addr + len < addr)
        return false;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type != unwind_pt_load)
                continue;
            uintptr_t start = 0;
            uintptr_t end = 0;
            unwind_load_range (info, ph, start, end);
            if (addr >= start && addr + len <= end)
                return true;
        }
    return false;
}

static bool
unwind_sections_eh_frame (uintptr_t bias, const ::dl_phdr_info *info,
                          uintptr_t &out_eh_frame, uintptr_t &out_eh_frame_len,
                          uintptr_t &out_hdr, uintptr_t &out_hdr_len) noexcept
{
    uintptr_t ehdr_addr = 0;
    bool found_ehdr = false;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type == unwind_pt_load && ph.p_offset == 0)
                {
                    ehdr_addr = bias + uintptr_t (ph.p_vaddr);
                    found_ehdr = true;
                    break;
                }
        }
    if (!found_ehdr || ehdr_addr < 4096)
        return false;
    const auto *ehdr = reinterpret_cast<const uint8_t *> (ehdr_addr);
    if (ehdr[0] != 0x7F || ehdr[1] != 'E' || ehdr[2] != 'L' || ehdr[3] != 'F')
        return false;
    const uint64_t shoff = unwind_le64 (ehdr + 40);
    const uint16_t shentsize = unwind_le16 (ehdr + 58);
    const uint16_t shnum = unwind_le16 (ehdr + 60);
    const uint16_t shstrndx = unwind_le16 (ehdr + 62);
    if (shentsize < 64 || shnum == 0 || shnum > 4096)
        return false;
    if (shoff > SIZE_MAX - ehdr_addr)
        return false;
    const uintptr_t shdrs_addr = ehdr_addr + uintptr_t (shoff);
    const size_t shdrs_size = size_t (shentsize) * size_t (shnum);
    if (!unwind_in_load_range (info, shdrs_addr, shdrs_size))
        return false;
    const auto *shdrs = reinterpret_cast<const uint8_t *> (shdrs_addr);
    const auto shdr_at = [&] (size_t index) noexcept -> const uint8_t *
        {
            const size_t off = index * size_t (shentsize);
            if (off + 64 > shdrs_size)
                return nullptr;
            return shdrs + off;
        };
    const uint8_t *strtab = shdr_at (shstrndx);
    if (strtab == nullptr)
        return false;
    const uintptr_t str_addr = bias + uintptr_t (unwind_le64 (strtab + 16));
    const size_t str_size = size_t (unwind_le64 (strtab + 32));
    if (str_addr < 4096 || str_size > (size_t{1} << 24)
        || !unwind_in_load_range (info, str_addr, str_size))
        return false;
    const auto *strtab_bytes = reinterpret_cast<const uint8_t *> (str_addr);

    bool has_eh_frame = false;
    uintptr_t eh_frame = 0;
    uintptr_t eh_frame_len = 0;
    uintptr_t hdr = 0;
    uintptr_t hdr_len = 0;
    for (size_t i = 0; i < shnum; ++i)
        {
            const uint8_t *sh = shdr_at (i);
            if (sh == nullptr)
                return false;
            const uint32_t name_off = unwind_le32 (sh);
            const uintptr_t sh_addr =
                bias + static_cast<uintptr_t> (unwind_le64 (sh + 16));
            const size_t sh_size = static_cast<size_t> (unwind_le64 (sh + 32));
            if (name_off >= str_size)
                continue;
            const char *name =
                reinterpret_cast<const char *> (strtab_bytes + name_off);
            size_t name_len = 0;
            while (name_len + name_off < str_size && name[name_len] != '\0')
                ++name_len;
            const bool is_eh_frame = name_len == 9 && name[0] == '.'
                                     && name[1] == 'e' && name[2] == 'h'
                                     && name[3] == '_' && name[4] == 'f'
                                     && name[5] == 'r' && name[6] == 'a'
                                     && name[7] == 'm' && name[8] == 'e';
            const bool is_hdr =
                name_len == 14 && name[0] == '.' && name[13] == 'e'
                && name[2] == 'h' && name[3] == '_' && name[4] == 'f'
                && name[5] == 'r' && name[6] == 'a' && name[7] == 'm'
                && name[8] == 'e' && name[9] == '_' && name[10] == 'h'
                && name[11] == 'd' && name[12] == 'r';
            if (is_eh_frame && sh_size > 0 && sh_size < (size_t{1} << 28))
                {
                    has_eh_frame = true;
                    eh_frame = sh_addr;
                    eh_frame_len = static_cast<uintptr_t> (sh_size);
                }
            else if (is_hdr && sh_size > 0)
                {
                    hdr = sh_addr;
                    hdr_len = static_cast<uintptr_t> (sh_size);
                }
        }
    if (!has_eh_frame)
        return false;
    out_eh_frame = eh_frame;
    out_eh_frame_len = eh_frame_len;
    out_hdr = hdr;
    out_hdr_len = hdr_len;
    return true;
}

static int
unwind_elf_callback (::dl_phdr_info *info, size_t, void *data) noexcept
{
    if (info == nullptr || data == nullptr)
        return 0;
    auto *search = static_cast<UnwindSearch *> (data);
    uintptr_t lo = ~uintptr_t{0};
    uintptr_t hi = 0;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type != unwind_pt_load)
                continue;
            uintptr_t start = 0;
            uintptr_t end = 0;
            unwind_load_range (info, ph, start, end);
            if (start < lo)
                lo = start;
            if (end > hi)
                hi = end;
        }
    if (hi <= lo || search->_M_ip < lo || search->_M_ip >= hi)
        return 0;

    UnwindFound &found = search->found;
    found._M_has_base = true;
    found._M_base = lo;

    uintptr_t hdr = 0;
    uintptr_t hdr_len = 0;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type == unwind_pt_gnu_eh_frame)
                {
                    hdr = uintptr_t (info->dlpi_addr) + uintptr_t (ph.p_vaddr);
                    hdr_len = uintptr_t (ph.p_memsz);
                    break;
                }
        }

    uintptr_t eh_frame = 0;
    uintptr_t eh_frame_len = 0;
    uintptr_t shdr_hdr = 0;
    uintptr_t shdr_hdr_len = 0;
    if (unwind_sections_eh_frame (uintptr_t (info->dlpi_addr), info, eh_frame,
                                  eh_frame_len, shdr_hdr, shdr_hdr_len))
        {
            found._M_has_eh_frame = true;
            found._M_eh_frame = eh_frame;
            found._M_eh_frame_len = eh_frame_len;
        }
    if (hdr == 0)
        {
            hdr = shdr_hdr;
            hdr_len = shdr_hdr_len;
        }
    if (hdr != 0 && hdr_len != 0)
        {
            found._M_has_hdr = true;
            found._M_hdr = hdr;
            found._M_hdr_len = hdr_len;
        }
    if (!found._M_has_eh_frame && found._M_has_hdr)
        {
            uintptr_t decoded = 0;
            uintptr_t decoded_len = 0;
            if (unwind_hdr_eh_frame (reinterpret_cast<const uint8_t *> (hdr),
                                     size_t (hdr_len), hdr, info, decoded,
                                     decoded_len))
                {
                    found._M_has_eh_frame = true;
                    found._M_eh_frame = decoded;
                    found._M_eh_frame_len = decoded_len;
                }
        }
    return 1;
}

static UnwindFound
unwind_elf_find (const uintptr_t ip) noexcept
{
    UnwindSearch search;
    search._M_ip = ip;
    search.found = UnwindFound{};
    ::dl_iterate_phdr (unwind_elf_callback, &search);
    return search.found;
}

static bool
unwind_elf_step (UnwindState &state) noexcept
{
    const UnwindFound found = unwind_elf_find (state._M_regs._M_ip);
    if (found._M_has_eh_frame)
        {
            UnwindTables tables;
            tables._M_eh_frame = found._M_eh_frame;
            tables._M_eh_frame_len = size_t (found._M_eh_frame_len);
            tables._M_has_hdr = found._M_has_hdr;
            tables._M_hdr = found._M_hdr;
            tables._M_hdr_len = size_t (found._M_hdr_len);
            tables._M_datarel_base = found._M_datarel_base;
            UnwindState next;
            const UnwindStep result = unwind_cfi_step (
                state, tables,
                reinterpret_cast<const uint8_t *> (found._M_eh_frame),
                size_t (found._M_eh_frame_len), next);
            if (result == UnwindStep::advanced)
                {
                    state = next;
                    return true;
                }
            if (result == UnwindStep::stop)
                return false;
        }
    return unwind_fp_step (state);
}

static int
unwind_cache_callback (::dl_phdr_info *info, size_t, void *data) noexcept
{
    if (info == nullptr || data == nullptr)
        return 1;
    auto *index = static_cast<size_t *> (data);
    if (*index >= _S_unwind_max_cached_modules)
        return 1;
    uintptr_t lo = ~uintptr_t{0};
    uintptr_t hi = 0;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type != unwind_pt_load)
                continue;
            uintptr_t start = 0;
            uintptr_t end = 0;
            unwind_load_range (info, ph, start, end);
            if (start < lo)
                lo = start;
            if (end > hi)
                hi = end;
        }
    if (hi <= lo)
        return 0;
    const char *name = info->dlpi_name;
    unwind_cache_store (*index, lo, hi,
                        (name != nullptr && name[0] != '\0') ? name : nullptr);
    ++*index;
    return 0;
}

static void
unwind_cache_build () noexcept
{
    size_t index = 0;
    ::dl_iterate_phdr (unwind_cache_callback, &index);
    unwind_cache_sort (index);
    unwind_module_count.store (index, std::memory_order_release);
}

static int
unwind_live_callback (::dl_phdr_info *info, size_t, void *data) noexcept
{
    if (info == nullptr || data == nullptr)
        return 0;
    auto *search = static_cast<UnwindSearch *> (data);
    uintptr_t lo = ~uintptr_t{0};
    uintptr_t hi = 0;
    for (size_t i = 0; i < info->dlpi_phnum; ++i)
        {
            const ElfW (Phdr) &ph = info->dlpi_phdr[i];
            if (ph.p_type != unwind_pt_load)
                continue;
            uintptr_t start = 0;
            uintptr_t end = 0;
            unwind_load_range (info, ph, start, end);
            if (start < lo)
                lo = start;
            if (end > hi)
                hi = end;
        }
    if (hi <= lo || search->_M_ip < lo || search->_M_ip >= hi)
        return 0;
    search->found._M_has_base = true;
    search->found._M_base = lo;
    return 1;
}
#endif

#if defined(ZZS_UNWIND_DYLD)
static bool
unwind_sect_name_eq (const char *sect, const char *want) noexcept
{
    size_t i = 0;
    while (i < 16 && want[i] != '\0')
        {
            if (sect[i] != want[i])
                return false;
            ++i;
        }
    if (want[i] != '\0')
        return false;
    if (i < 16 && sect[i] != '\0' && sect[i] != ' ')
        return false;
    return true;
}

static bool
unwind_dyld_image_info (uintptr_t ip, uintptr_t &out_base,
                        uintptr_t &out_eh_frame, size_t &out_eh_frame_len,
                        uintptr_t &out_hdr, size_t &out_hdr_len) noexcept
{
    out_base = 0;
    out_eh_frame = 0;
    out_eh_frame_len = 0;
    out_hdr = 0;
    out_hdr_len = 0;
#if defined(__LP64__)
    const uint32_t image_count = ::_dyld_image_count ();
    for (uint32_t i = 0; i < image_count; ++i)
        {
            const auto *mh = reinterpret_cast<const ::mach_header_64 *> (
                ::_dyld_get_image_header (i));
            if (mh == nullptr || mh->magic != MH_MAGIC_64)
                continue;
            const intptr_t slide = ::_dyld_get_image_vmaddr_slide (i);
            const auto *cursor = reinterpret_cast<const uint8_t *> (mh)
                                 + sizeof (::mach_header_64);
            uintptr_t lo = ~uintptr_t{0};
            uintptr_t hi = 0;
            uintptr_t eh_frame = 0;
            size_t eh_frame_len = 0;
            uintptr_t hdr = 0;
            size_t hdr_len = 0;
            for (uint32_t c = 0; c < mh->ncmds; ++c)
                {
                    const auto *lc =
                        reinterpret_cast<const ::load_command *> (cursor);
                    if (lc->cmd == LC_SEGMENT_64)
                        {
                            const auto *seg =
                                reinterpret_cast<const ::segment_command_64 *> (
                                    cursor);
                            const uintptr_t start =
                                uintptr_t (seg->vmaddr) + uintptr_t (slide);
                            const uintptr_t end =
                                start + uintptr_t (seg->vmsize);
                            if (start < lo)
                                lo = start;
                            if (end > hi)
                                hi = end;
                            const auto *sections =
                                reinterpret_cast<const ::section_64 *> (
                                    cursor + sizeof (::segment_command_64));
                            for (uint32_t s = 0; s < seg->nsects; ++s)
                                {
                                    const ::section_64 &sec = sections[s];
                                    const uintptr_t addr = uintptr_t (sec.addr)
                                                           + uintptr_t (slide);
                                    if (eh_frame == 0
                                        && unwind_sect_name_eq (sec.sectname,
                                                                "__eh_frame")
                                        && unwind_sect_name_eq (sec.segname,
                                                                "__TEXT"))
                                        {
                                            eh_frame = addr;
                                            eh_frame_len = size_t (sec.size);
                                        }
                                    else if (hdr == 0
                                             && unwind_sect_name_eq (
                                                 sec.sectname, "__eh_frame_hdr")
                                             && (unwind_sect_name_eq (
                                                     sec.segname, "__DATA")
                                                 || unwind_sect_name_eq (
                                                     sec.segname,
                                                     "__DATA_CONST")))
                                        {
                                            hdr = addr;
                                            hdr_len = size_t (sec.size);
                                        }
                                }
                        }
                    cursor += lc->cmdsize;
                }
            if (hi <= lo || ip < lo || ip >= hi)
                continue;
            out_base = lo;
            out_eh_frame = eh_frame;
            out_eh_frame_len = eh_frame_len;
            out_hdr = hdr;
            out_hdr_len = hdr_len;
            return true;
        }
#endif
    return false;
}

static bool
unwind_dyld_step (UnwindState &state) noexcept
{
    uintptr_t base = 0;
    uintptr_t eh_frame = 0;
    size_t eh_frame_len = 0;
    uintptr_t hdr = 0;
    size_t hdr_len = 0;
    if (unwind_dyld_image_info (state._M_regs._M_ip, base, eh_frame,
                                eh_frame_len, hdr, hdr_len)
        && eh_frame != 0)
        {
            UnwindTables tables;
            tables._M_eh_frame = eh_frame;
            tables._M_eh_frame_len = eh_frame_len;
            tables._M_has_hdr = hdr != 0;
            tables._M_hdr = hdr;
            tables._M_hdr_len = hdr_len;
            UnwindState next;
            const UnwindStep result = unwind_cfi_step (
                state, tables, reinterpret_cast<const uint8_t *> (eh_frame),
                eh_frame_len, next);
            if (result == UnwindStep::advanced)
                {
                    state = next;
                    return true;
                }
            if (result == UnwindStep::stop)
                return false;
        }
    return unwind_fp_step (state);
}
#endif

#if defined(ZZS_UNWIND_RTL)
static bool
unwind_windows_step (UnwindState &state) noexcept
{
#if defined(_M_X64) || defined(__x86_64__)
    const UnwindRegs &regs = state._M_regs;
    ::CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    ctx.Rax = regs._M_gpr[0];
    ctx.Rdx = regs._M_gpr[1];
    ctx.Rcx = regs._M_gpr[2];
    ctx.Rbx = regs._M_gpr[3];
    ctx.Rsi = regs._M_gpr[4];
    ctx.Rdi = regs._M_gpr[5];
    ctx.Rbp = regs._M_gpr[6];
    ctx.Rsp = regs._M_gpr[7];
    ctx.R8 = regs._M_gpr[8];
    ctx.R9 = regs._M_gpr[9];
    ctx.R10 = regs._M_gpr[10];
    ctx.R11 = regs._M_gpr[11];
    ctx.R12 = regs._M_gpr[12];
    ctx.R13 = regs._M_gpr[13];
    ctx.R14 = regs._M_gpr[14];
    ctx.R15 = regs._M_gpr[15];
    ctx.Rip = regs._M_ip;
    ::DWORD64 image_base = 0;
    ::PRUNTIME_FUNCTION function =
        ::RtlLookupFunctionEntry (ctx.Rip, &image_base, nullptr);
    if (function == nullptr)
        return unwind_fp_step (state);
    void *handler_data = nullptr;
    ::DWORD64 establisher_frame = 0;
    ::RtlVirtualUnwind (UNW_FLAG_NHANDLER, image_base, ctx.Rip, function, &ctx,
                        &handler_data, &establisher_frame, nullptr);
    if (ctx.Rip == 0 || ctx.Rsp <= regs._M_gpr[7])
        return false;
    UnwindState next;
    next._M_regs = state._M_regs;
    next._M_regs._M_gpr[0] = uintptr_t (ctx.Rax);
    next._M_regs._M_gpr[1] = uintptr_t (ctx.Rdx);
    next._M_regs._M_gpr[2] = uintptr_t (ctx.Rcx);
    next._M_regs._M_gpr[3] = uintptr_t (ctx.Rbx);
    next._M_regs._M_gpr[4] = uintptr_t (ctx.Rsi);
    next._M_regs._M_gpr[5] = uintptr_t (ctx.Rdi);
    next._M_regs._M_gpr[6] = uintptr_t (ctx.Rbp);
    next._M_regs._M_gpr[7] = uintptr_t (ctx.Rsp);
    next._M_regs._M_gpr[8] = uintptr_t (ctx.R8);
    next._M_regs._M_gpr[9] = uintptr_t (ctx.R9);
    next._M_regs._M_gpr[10] = uintptr_t (ctx.R10);
    next._M_regs._M_gpr[11] = uintptr_t (ctx.R11);
    next._M_regs._M_gpr[12] = uintptr_t (ctx.R12);
    next._M_regs._M_gpr[13] = uintptr_t (ctx.R13);
    next._M_regs._M_gpr[14] = uintptr_t (ctx.R14);
    next._M_regs._M_gpr[15] = uintptr_t (ctx.R15);
    next._M_regs._M_gpr[16] = uintptr_t (ctx.Rip);
    next._M_regs._M_ip = uintptr_t (ctx.Rip);
    next._M_sp = uintptr_t (ctx.Rsp);
    state = next;
    return true;
#elif defined(_M_ARM64) || defined(__aarch64__)
    const UnwindRegs &regs = state._M_regs;
    ::CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    ctx.X0 = regs._M_gpr[0];
    ctx.X1 = regs._M_gpr[1];
    ctx.X2 = regs._M_gpr[2];
    ctx.X3 = regs._M_gpr[3];
    ctx.X4 = regs._M_gpr[4];
    ctx.X5 = regs._M_gpr[5];
    ctx.X6 = regs._M_gpr[6];
    ctx.X7 = regs._M_gpr[7];
    ctx.X8 = regs._M_gpr[8];
    ctx.X9 = regs._M_gpr[9];
    ctx.X10 = regs._M_gpr[10];
    ctx.X11 = regs._M_gpr[11];
    ctx.X12 = regs._M_gpr[12];
    ctx.X13 = regs._M_gpr[13];
    ctx.X14 = regs._M_gpr[14];
    ctx.X15 = regs._M_gpr[15];
    ctx.X16 = regs._M_gpr[16];
    ctx.X17 = regs._M_gpr[17];
    ctx.X18 = regs._M_gpr[18];
    ctx.X19 = regs._M_gpr[19];
    ctx.X20 = regs._M_gpr[20];
    ctx.X21 = regs._M_gpr[21];
    ctx.X22 = regs._M_gpr[22];
    ctx.X23 = regs._M_gpr[23];
    ctx.X24 = regs._M_gpr[24];
    ctx.X25 = regs._M_gpr[25];
    ctx.X26 = regs._M_gpr[26];
    ctx.X27 = regs._M_gpr[27];
    ctx.X28 = regs._M_gpr[28];
    ctx.Fp = regs._M_gpr[29];
    ctx.Lr = regs._M_gpr[30];
    ctx.Sp = regs._M_gpr[31];
    ctx.Pc = regs._M_ip;
    ::DWORD64 image_base = 0;
    ::PRUNTIME_FUNCTION function =
        ::RtlLookupFunctionEntry (ctx.Pc, &image_base, nullptr);
    if (function == nullptr)
        return unwind_fp_step (state);
    void *handler_data = nullptr;
    ::DWORD64 establisher_frame = 0;
    ::RtlVirtualUnwind (UNW_FLAG_NHANDLER, image_base, ctx.Pc, function, &ctx,
                        &handler_data, &establisher_frame, nullptr);
    if (ctx.Pc == 0 || ctx.Sp <= regs._M_gpr[31])
        return false;
    UnwindState next;
    next._M_regs = state._M_regs;
    next._M_regs._M_gpr[0] = uintptr_t (ctx.X0);
    next._M_regs._M_gpr[1] = uintptr_t (ctx.X1);
    next._M_regs._M_gpr[2] = uintptr_t (ctx.X2);
    next._M_regs._M_gpr[3] = uintptr_t (ctx.X3);
    next._M_regs._M_gpr[4] = uintptr_t (ctx.X4);
    next._M_regs._M_gpr[5] = uintptr_t (ctx.X5);
    next._M_regs._M_gpr[6] = uintptr_t (ctx.X6);
    next._M_regs._M_gpr[7] = uintptr_t (ctx.X7);
    next._M_regs._M_gpr[8] = uintptr_t (ctx.X8);
    next._M_regs._M_gpr[9] = uintptr_t (ctx.X9);
    next._M_regs._M_gpr[10] = uintptr_t (ctx.X10);
    next._M_regs._M_gpr[11] = uintptr_t (ctx.X11);
    next._M_regs._M_gpr[12] = uintptr_t (ctx.X12);
    next._M_regs._M_gpr[13] = uintptr_t (ctx.X13);
    next._M_regs._M_gpr[14] = uintptr_t (ctx.X14);
    next._M_regs._M_gpr[15] = uintptr_t (ctx.X15);
    next._M_regs._M_gpr[16] = uintptr_t (ctx.X16);
    next._M_regs._M_gpr[17] = uintptr_t (ctx.X17);
    next._M_regs._M_gpr[18] = uintptr_t (ctx.X18);
    next._M_regs._M_gpr[19] = uintptr_t (ctx.X19);
    next._M_regs._M_gpr[20] = uintptr_t (ctx.X20);
    next._M_regs._M_gpr[21] = uintptr_t (ctx.X21);
    next._M_regs._M_gpr[22] = uintptr_t (ctx.X22);
    next._M_regs._M_gpr[23] = uintptr_t (ctx.X23);
    next._M_regs._M_gpr[24] = uintptr_t (ctx.X24);
    next._M_regs._M_gpr[25] = uintptr_t (ctx.X25);
    next._M_regs._M_gpr[26] = uintptr_t (ctx.X26);
    next._M_regs._M_gpr[27] = uintptr_t (ctx.X27);
    next._M_regs._M_gpr[28] = uintptr_t (ctx.X28);
    next._M_regs._M_gpr[29] = uintptr_t (ctx.Fp);
    next._M_regs._M_gpr[30] = uintptr_t (ctx.Lr);
    next._M_regs._M_gpr[31] = uintptr_t (ctx.Sp);
    next._M_regs._M_ip = uintptr_t (ctx.Pc);
    next._M_sp = uintptr_t (ctx.Sp);
    state = next;
    return true;
#else
    return unwind_fp_step (state);
#endif
}
#endif

bool
unwind_step_platform (UnwindState &state) noexcept
{
#if defined(ZZS_UNWIND_ELF)
    return unwind_elf_step (state);
#elif defined(ZZS_UNWIND_DYLD)
    return unwind_dyld_step (state);
#elif defined(ZZS_UNWIND_RTL)
    return unwind_windows_step (state);
#else
    return unwind_fp_step (state);
#endif
}

static uintptr_t
unwind_live_module_base (uintptr_t ip) noexcept
{
#if defined(ZZS_UNWIND_ELF)
    UnwindSearch search;
    search._M_ip = ip;
    search.found = UnwindFound{};
    ::dl_iterate_phdr (unwind_live_callback, &search);
    return search.found._M_has_base ? search.found._M_base : 0;
#elif defined(ZZS_UNWIND_DYLD)
    uintptr_t base = 0;
    uintptr_t eh_frame = 0;
    size_t eh_frame_len = 0;
    uintptr_t hdr = 0;
    size_t hdr_len = 0;
    if (unwind_dyld_image_info (ip, base, eh_frame, eh_frame_len, hdr, hdr_len))
        return base;
    return 0;
#elif defined(ZZS_UNWIND_RTL)
    ::MEMORY_BASIC_INFORMATION info{};
    if (::VirtualQuery (reinterpret_cast<const void *> (ip), &info,
                        sizeof (info))
        == 0)
        return 0;
    return reinterpret_cast<uintptr_t> (info.AllocationBase);
#else
    (void) ip;
    return 0;
#endif
}

inline void
unwind_init_modules () noexcept
{
    if (unwind_modules_ready.load (std::memory_order_acquire))
        return;
#if defined(ZZS_UNWIND_ELF)
    unwind_cache_build ();
#else
    unwind_modules_ready.store (true, std::memory_order_release);
#endif
}

inline Maybe<uintptr_t, FrameUnwindError>
unwind_module_base (uintptr_t ip) noexcept
{
    unwind_init_modules ();
    const UnwindModuleEntry *entry = unwind_cache_lookup (ip);
    if (entry != nullptr)
        return entry->_M_base;
    const uintptr_t base = unwind_live_module_base (ip);
    if (base == 0)
        return Maybe<uintptr_t, FrameUnwindError> (
            FrameUnwindError (FrameUnwindError::Kind::INVALID_ADDR));
    return base;
}

SimpleStringView
unwind_module_name (uintptr_t base) noexcept
{
    unwind_init_modules ();
    const size_t count = unwind_module_count.load (std::memory_order_acquire);
    for (size_t i = 0; i < count; ++i)
        {
            const UnwindModuleEntry &entry = unwind_module_cache[i];
            if (entry._M_base == base && entry._M_name[0] != '\0')
                return SimpleStringView (entry._M_name);
        }
        // dl_iterate_phdr reports an empty dlpi_name for the main
        // executable; dladdr resolves it to the invocation path.
#if defined(ZZS_UNWIND_ELF) || defined(ZZS_UNWIND_DYLD)
    Dl_info info{};
    if (::dladdr (reinterpret_cast<const void *> (base), &info) != 0
        && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
        return SimpleStringView (info.dli_fname);
#endif
    return SimpleStringView{};
}

template <typename Callback>
void
unwind_walk_impl (size_t limit, Callback &&cb) noexcept
{
    if (limit == 0)
        return;
    UnwindState state;
    if (!unwind_capture_current (state))
        return;
    unwind_init_modules ();
    size_t index = 0;
    uintptr_t prev_ip = state._M_regs._M_ip;
    uintptr_t prev_sp = state._M_sp;
    while (index < limit && state._M_regs._M_ip != 0)
        {
            UnwindFrame frame;
            frame._M_ip = state._M_regs._M_ip;
            frame._M_sp = state._M_sp;
            frame._M_module_base = unwind_module_base (state._M_regs._M_ip);
            cb (frame, index);
            ++index;
            if (state._M_stop)
                {
                    break;
                }
            const bool stepped = unwind_step_platform (state);
            if (!stepped)
                {
                    break;
                }
            if (state._M_regs._M_ip == prev_ip || state._M_sp <= prev_sp)
                {
                    break;
                }
            prev_ip = state._M_regs._M_ip;
            prev_sp = state._M_sp;
        }
}

template <typename Callback>
void
unwind_trace (Callback &&cb) noexcept
{
    unwind_walk_impl (_S_unwind_max_frames, static_cast<Callback &&> (cb));
}

size_t
unwind_capture_frames (UnwindFrame *out, size_t cap) noexcept
{
    if (out == nullptr)
        return 0;
    if (cap != 0)
        {
            const size_t limit =
                cap < _S_unwind_max_frames ? cap : _S_unwind_max_frames;
            size_t count = 0;
            unwind_walk_impl (
                limit,
                [&out, &cap, &count] (const UnwindFrame &frame, size_t) noexcept
                    {
                        if (count < cap)
                            out[count++] = frame;
                    });
            return count;
        }
    else
        return 0;
}
} // namespace zzs::base