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

#include "std/basic_iterator.hpp"
#include "std/fmt_sink.hpp"
#include "std/maybe.hpp"
#include "std/string_view.hpp"
#include "std/sunwind.hpp"

#include <cinttypes>
#include <concepts>
#include <dlfcn.h>
#include <memory>
#include <new>
#include <signal.h>
#include <span>
#include <type_traits>

namespace zzs::base
{
/*
 * Resolves the nearest symbol name for addr via dladdr.
 *
 * Returns the mangled symbol name when dladdr finds one, the module
 * path when the address belongs to a module without a symbol, and an
 * empty view when the address cannot be resolved at all.
 */
inline SimpleStringView
resolve_symbol (const uintptr_t addr) noexcept
{
    (void) addr;
    return SimpleStringView{};
    Dl_info info{};
    if (::dladdr (reinterpret_cast<const void *> (addr), &info) == 0)
        {
            return SimpleStringView{};
        }
    if (info.dli_sname != nullptr && info.dli_sname[0] != '\0')
        {
            return SimpleStringView (info.dli_sname);
        }
    if (info.dli_fname != nullptr)
        {
            return SimpleStringView (info.dli_fname);
        }
    return SimpleStringView{};
}

/* Resolves the module path for addr via dladdr; empty view when unresolved. */
inline SimpleStringView
resolve_module (const uintptr_t addr) noexcept
{
    Dl_info info{};
    if (::dladdr (reinterpret_cast<const void *> (addr), &info) == 0)
        {
            return SimpleStringView{};
        }
    if (info.dli_fname != nullptr)
        {
            return SimpleStringView (info.dli_fname);
        }
    return SimpleStringView{};
}

/*
 * Formats a single frame as:
 * "   {index:>3}: ip=0x{ip:016x} sp=0x{sp:016x}[+0x{offset:x}] [<(sym)>]
 * [(mod)]"
 *
 * The offset and symbol parts appear only when the frame carries a
 * module base; the module part is resolved independently.
 */
template <typename Sink>
void
write_frame (Sink &sink, const UnwindFrame &frame, const size_t index) noexcept
{
    sink.write ("   ");
    write_uint (sink, index, 3);
    sink.write (": ip=0x");
    write_hex (sink, frame._M_ip, 16);
    sink.write (" sp=0x");
    write_hex (sink, frame._M_sp, 16);
    if (frame._M_module_base.has_value ())
        {
            const uintptr_t base = frame._M_module_base.value ();
            sink.write ("+0x");
            write_hex (sink, frame._M_ip - base, 0);
            const SimpleStringView symbol = resolve_symbol (frame._M_ip);
            if (!symbol.empty ())
                {
                    sink.write (" <");
                    sink.write (symbol);
                    sink.write (">");
                }
        }
    const SimpleStringView module = resolve_module (frame._M_ip);
    if (!module.empty ())
        {
            sink.write (" (");
            sink.write (module);
            sink.write (")");
        }
}

/* Formats frames as '#' prefixed lines separated by '\n' (no trailing newline).
 */
template <typename Sink>
void
write_frames (Sink &sink, const std::span<const UnwindFrame> frames) noexcept
{
    for (size_t i = 0; i < frames.size (); ++i)
        {
            if (i > 0)
                sink.write ("\n");
            sink.write ("#");
            write_frame (sink, frames[i], i);
        }
}

/* Writes the frame's symbol address as a 16-digit hex string. */
template <typename Sink>
void
write_symbol_address (Sink &sink, const UnwindFrame &frame) noexcept
{
    write_hex (sink, frame.symbol_address (), 16);
}

enum class SigInstallErrorKind
{
    UNSUPPORTED,
    SYSCALL
};

struct SigInstallError
{
    SigInstallErrorKind _M_kind = SigInstallErrorKind::UNSUPPORTED;
    const char *_M_op = "";
    int _M_errno_value = 0;
};
/* Maximum signals tracked by install/uninstall (23 used today). */
inline constexpr size_t _S_sig_max_saved = 0x20;

/* Alternate signal stack installed by sig_install. */
inline constexpr size_t _S_sig_default_alt_stack_size = 0x10000;

/* Minimum accepted size for a user-supplied alternate signal stack. */
inline constexpr size_t _S_sig_min_alt_stack_size = 0x2000;

/* Maximum frames captured by a backtrace dump. */
inline constexpr size_t _S_sig_dump_frames_max = 0x40;

static std::atomic sig_installed{false};
/* Returns a short symbolic name for sig, or "SIGUNKNOWN". */
[[nodiscard]] inline SimpleStringView
sig_signal_name (const int sig) noexcept
{
#if defined(_WIN32)
    (void) sig;
    return "SIGUNKNOWN";
#else
    switch (sig)
        {
        case SIGHUP:
            return "SIGHUP";
        case SIGINT:
            return "SIGINT";
        case SIGQUIT:
            return "SIGQUIT";
        case SIGILL:
            return "SIGILL";
        case SIGTRAP:
            return "SIGTRAP";
        case SIGABRT:
            return "SIGABRT";
        case SIGBUS:
            return "SIGBUS";
        case SIGFPE:
            return "SIGFPE";
        case SIGKILL:
            return "SIGKILL";
        case SIGUSR1:
            return "SIGUSR1";
        case SIGSEGV:
            return "SIGSEGV";
        case SIGUSR2:
            return "SIGUSR2";
        case SIGPIPE:
            return "SIGPIPE";
        case SIGALRM:
            return "SIGALRM";
        case SIGTERM:
            return "SIGTERM";
        case SIGCHLD:
            return "SIGCHLD";
        case SIGCONT:
            return "SIGCONT";
        case SIGSTOP:
            return "SIGSTOP";
        case SIGTSTP:
            return "SIGTSTP";
        case SIGTTIN:
            return "SIGTTIN";
        case SIGTTOU:
            return "SIGTTOU";
        case SIGURG:
            return "SIGURG";
        case SIGXCPU:
            return "SIGXCPU";
        case SIGXFSZ:
            return "SIGXFSZ";
        case SIGVTALRM:
            return "SIGVTALRM";
        case SIGPROF:
            return "SIGPROF";
        case SIGWINCH:
            return "SIGWINCH";
        case SIGSYS:
            return "SIGSYS";
#ifdef SIGSTKFLT
        case SIGSTKFLT:
            return "SIGSTKFLT";
#endif
#ifdef SIGPWR
        case SIGPWR:
            return "SIGPWR";
#endif
#ifdef SIGIO
        case SIGIO:
            return "SIGIO";
#endif
        default:
            return "SIGUNKNOWN";
        }
#endif
}

/*
 * Returns true for synchronous fault signals (where si_addr is
 * meaningful and the faulting instruction pointer is worth
 * reporting).
 */
[[nodiscard]] inline bool
sig_is_fault_signal (const int sig) noexcept
{
#if defined(_WIN32)
    (void) sig;
    return false;
#else
    switch (sig)
        {
        case SIGILL:
        case SIGTRAP:
        case SIGABRT:
        case SIGBUS:
        case SIGFPE:
        case SIGSEGV:
        case SIGSYS:
        case SIGXCPU:
        case SIGXFSZ:
            return true;
#ifdef SIGSTKFLT
        case SIGSTKFLT:
            return true;
#endif
        default:
            return false;
        }
#endif
}

/*
 * Formats err into buf without a trailing newline, truncating to
 * cap - 1 bytes plus the terminator. Returns the byte count written
 * (excluding the terminator); buf must hold at least cap bytes.
 */
[[nodiscard]] inline size_t
sig_error_message (const SigInstallError &err, char *const buf,
                   const size_t cap) noexcept
{
    if (buf == nullptr || cap == 0)
        return 0;
    FrameSink<256> line;
    if (err._M_kind == SigInstallErrorKind::UNSUPPORTED)
        line.write ("<signal handling not supported on this target>");
    else
        {
            line.write ("`");
            if (err._M_op != nullptr)
                line.write (err._M_op);
            line.write ("` failed with errno ");
            long long value = err._M_errno_value;
            if (value < 0)
                {
                    line.write ("-");
                    value = -value;
                }
            write_uint (line, static_cast<uint64_t> (value), 0);
        }
    const SimpleStringView text = line.view ();
    const size_t count = text.size () < cap - 1 ? text.size () : cap - 1;
    for (size_t i = 0; i < count; ++i)
        buf[i] = text.data ()[i];
    buf[count] = '\0';
    return count;
}

/* True between a successful sig_install and the matching sig_uninstall. */
[[nodiscard]] inline bool
sig_is_installed () noexcept
{
    return sig_installed.load (std::memory_order_acquire);
}

/* True while executing inside a dump or signal handler. */
[[nodiscard]] inline bool
sig_is_in_handler () noexcept
{
    return unwind_in_handler ();
}

#if !defined(_WIN32)
/* Signals covered by sig_install (fault + termination + job control). */
static int sig_handled_signals[] = {
    SIGILL,  SIGTRAP, SIGABRT, SIGBUS,  SIGFPE,    SIGSEGV,
    SIGSYS,  SIGXCPU, SIGXFSZ, SIGHUP,  SIGINT,    SIGQUIT,
    SIGTERM, SIGUSR1, SIGUSR2, SIGALRM, SIGVTALRM, SIGPROF,
#ifdef SIGPWR
    SIGPWR,
#endif
#ifdef SIGIO
    SIGIO,
#endif
    SIGTSTP, SIGTTIN, SIGTTOU,
};

static constexpr size_t _S_sig_handled_count = std::size (sig_handled_signals);
static_assert (_S_sig_handled_count <= _S_sig_max_saved);

/*
 * Saved previous dispositions for sig_uninstall. Touched only from
 * install/uninstall context (never from a signal handler); callers
 * must not race those entry points.
 */
static int _S_sig_saved_signals[_S_sig_max_saved] = {};
static struct sigaction _S_sig_saved_actions[_S_sig_max_saved] = {};
static size_t _S_sig_saved_count = 0;

/* Default alternate signal stack (16-byte aligned, process lifetime). */
alignas (
    16) static char _S_sig_default_alt_stack[_S_sig_default_alt_stack_size] =
    {};

static void
sig_save_prev (const int sig, const struct sigaction &act) noexcept
{
    if (_S_sig_saved_count < _S_sig_max_saved)
        {
            _S_sig_saved_signals[_S_sig_saved_count] = sig;
            _S_sig_saved_actions[_S_sig_saved_count] = act;
            ++_S_sig_saved_count;
        }
}

/* Restores every saved disposition in reverse order and clears the table. */
static void
sig_restore_prev () noexcept
{
    for (size_t i = _S_sig_saved_count; i > 0; --i)
        ::sigaction (_S_sig_saved_signals[i - 1], &_S_sig_saved_actions[i - 1],
                     nullptr);
    _S_sig_saved_count = 0;
}

static void
sig_write_raw (const SimpleStringView text) noexcept
{
    size_t off = 0;
    while (off < text.size ())
        {
            const ssize_t written =
                ::write (STDERR_FILENO, text.data () + off, text.size () - off);
            if (written > 0)
                {
                    off += static_cast<size_t> (written);
                    continue;
                }
            if (written < 0 && errno == EINTR)
                continue;
            break;
        }
}

/* Appends a newline, writes the line to stderr and resets the sink. */
template <unsigned int N>
static void
sig_flush_line (FrameSink<N> &line) noexcept
{
    line.write ("\n");
    sig_write_raw (line.view ());
    line.clear ();
}

/* Resets sig to SIG_DFL, unblocks it and re-raises the default action. */
static void
sig_reset_and_raise (const int sig) noexcept
{
    struct sigaction dfl{};
    dfl.sa_handler = SIG_DFL;
    ::sigaction (sig, &dfl, nullptr);

    sigset_t set;
    ::sigemptyset (&set);
    ::sigaddset (&set, sig);
    ::sigprocmask (SIG_UNBLOCK, &set, nullptr);
    ::raise (sig);
}

/* Extracts the faulting instruction pointer from a kernel ucontext. */
static uintptr_t
sig_fault_ip (const void *const uc) noexcept
{
    if (uc == nullptr)
        return 0;
#if defined(__linux__) && defined(__x86_64__)
    const ucontext_t *const ctx = static_cast<const ucontext_t *> (uc);
    return static_cast<uintptr_t> (ctx->uc_mcontext.gregs[REG_RIP]);
#elif defined(__linux__) && defined(__aarch64__)
    const ucontext_t *const ctx = static_cast<const ucontext_t *> (uc);
    return static_cast<uintptr_t> (ctx->uc_mcontext.pc);
#else
    return 0;
#endif
}

static void
sig_dump_frames () noexcept
{
    UnwindFrame frames[_S_sig_dump_frames_max];
    const size_t count = unwind_capture_frames (frames, _S_sig_dump_frames_max);
    for (size_t i = 0; i < count; ++i)
        {
            FrameSink<512> line;
            write_frame (line, frames[i], i);
            sig_flush_line (line);
        }
}

extern "C" inline void
zzs_sig_handler (const int sig, siginfo_t *const info,
                      void *const uc) noexcept
{
    // Reentrancy: never wait, never nest; a held guard means a
    // recursive fault while another report was being written.
    if (!unwind_try_enter ())
        {
            sig_write_raw ("recursive signal during handling\n");
            sig_reset_and_raise (sig);
            return;
        }
    const uintptr_t addr = (sig_is_fault_signal (sig) && info != nullptr)
                               ? reinterpret_cast<uintptr_t> (info->si_addr)
                               : 0;
    FrameSink<512> line;
    line.write ("Caught signal ");
    write_uint (line, static_cast<uint64_t> (sig), 0);
    line.write (" (");
    line.write (sig_signal_name (sig));
    line.write (")");
    sig_flush_line (line);
    if (addr != 0)
        {
            line.write ("fault address: 0x");
            write_hex (line, addr, 16);
            sig_flush_line (line);
        }
    const uintptr_t ip = sig_fault_ip (uc);
    if (ip != 0)
        {
            line.write ("fault IP:      0x");
            write_hex (line, ip, 16);
            sig_flush_line (line);
        }
    line.write ("Backtrace: ");
    sig_flush_line (line);
    sig_dump_frames ();

    sig_reset_and_raise (sig);
    // Only reachable for job-control stops resumed with SIGCONT
    unwind_leave ();
}

static bool
sig_install_impl (SigInstallError *const err) noexcept
{
    stack_t ss{};
    ss.ss_sp = _S_sig_default_alt_stack;
    ss.ss_size = sizeof (_S_sig_default_alt_stack);
    ss.ss_flags = 0;
    if (::sigaltstack (&ss, nullptr) != 0)
        {
            if (err != nullptr)
                *err = SigInstallError{SigInstallErrorKind::SYSCALL,
                                       "sigaltstack", errno};
            return false;
        }

    struct sigaction act{};
    act.sa_sigaction = zzs_sig_handler;
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    ::sigemptyset (&act.sa_mask);
    for (size_t i = 0; i < _S_sig_handled_count; ++i)
        ::sigaddset (&act.sa_mask, sig_handled_signals[i]);
    // Fresh save table for this install attempt
    _S_sig_saved_count = 0;
    for (size_t i = 0; i < _S_sig_handled_count; ++i)
        {
            struct sigaction old{};
            if (::sigaction (sig_handled_signals[i], &act, &old) != 0)
                {
                    const int saved_errno = errno;
                    // Roll back to the dispositions seen before
                    // this attempt, not SIG_DFL: an unrelated
                    // handler must not be clobbered by a failure.
                    sig_restore_prev ();
                    if (err != nullptr)
                        *err = SigInstallError{SigInstallErrorKind::SYSCALL,
                                               "sigaction", saved_errno};
                    return false;
                }
            sig_save_prev (sig_handled_signals[i], old);
        }
    return true;
}

/*
 * Installs the crash-reporting handlers and the default alternate
 * signal stack. Returns true when installed (calling this again
 * while installed is a no-op). On failure err receives the
 * operation name and OS error; see SigInstallError.
 */
[[nodiscard]] inline bool
sig_install (SigInstallError *const err = nullptr) noexcept
{
    if (err != nullptr)
        {
            *err = SigInstallError{};
        }
    if (sig_installed.load (std::memory_order_acquire))
        {
            return true;
        }
    if (!sig_install_impl (err))
        {
            return false;
        }
    sig_installed.store (true, std::memory_order_release);
    return true;
}

/*
 * Restores the dispositions sig_install replaced. No-op when
 * nothing is installed; does not touch the alternate signal stack.
 */
[[nodiscard]] inline bool
sig_uninstall () noexcept
{
    if (!sig_installed.load (std::memory_order_acquire))
        return true;
    sig_restore_prev ();
    sig_installed.store (false, std::memory_order_release);
    return true;
}

/*
 * Sets buf as the alternate signal stack of the current thread.
 * buf must stay alive while installed; sizes below
 * _S_sig_min_alt_stack_size are rejected with errno EINVAL.
 */
[[nodiscard]] inline bool
sig_install_alt_stack (void *const buf, const size_t size,
                       SigInstallError *const err = nullptr) noexcept
{
    if (err != nullptr)
        *err = SigInstallError{};
    if (size < _S_sig_min_alt_stack_size)
        {
            if (err != nullptr)
                *err = SigInstallError{._M_kind = SigInstallErrorKind::SYSCALL,
                                       ._M_op = "sigaltstack:size",
                                       ._M_errno_value = EINVAL};
            return false;
        }
    stack_t ss{};
    ss.ss_sp = buf;
    ss.ss_size = size;
    ss.ss_flags = 0;
    if (::sigaltstack (&ss, nullptr) != 0)
        {
            if (err != nullptr)
                *err = SigInstallError{._M_kind = SigInstallErrorKind::SYSCALL,
                                       ._M_op = "sigaltstack",
                                       ._M_errno_value = errno};
            return false;
        }
    return true;
}

/*
 * Writes a backtrace of the current thread to stderr. Silent when
 * a dump or handler is already in progress (shared reentrancy
 * guard); never throws and never allocates.
 */
inline void
sig_dump_backtrace () noexcept
{
    if (!unwind_try_enter ())
        return;
    sig_dump_frames ();
    unwind_leave ();
}
#else
[[nodiscard]] bool
sig_install (SigInstallError *const = nullptr) noexcept
{
    return false;
}

[[nodiscard]] bool
sig_uninstall () noexcept
{
    return true;
}

[[nodiscard]] bool
sig_install_alt_stack (void *const, const size_t,
                       SigInstallError *const = nullptr) noexcept
{
    return false;
}
void
sig_dump_backtrace () noexcept
{
}
#endif
} // namespace zzs::base