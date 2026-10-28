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

#include "std/io.hpp"
#include "std/fmt_sink.hpp"
#include "std/basic_iterator.hpp"
#include "std/maybe.hpp"
#include "std/string_view.hpp"
#include "std/sunwind.hpp"

#include <cinttypes>
#include <concepts>
#include <memory>
#include <new>
#include <signal.h>
#include <type_traits>

namespace zzs::base
{
enum class LogLevel : uint8_t
{
    DEBUG = 0,
    INFO = 1,
    ERROR = 2,
    IRR = 3,
};

/* Returns the level name as it appears in log lines ("DEBUG", ...). */
constexpr SimpleStringView
as_str (const LogLevel level) noexcept
{
    switch (level)
        {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::IRR:
            return "IRR";
        }
    return "IRR";
}

/* Global minimum level; messages below it are dropped (relaxed atomics, like
 * Rust). */
static std::atomic<uint8_t> log_min_level{0};

inline void
set_min_log_level (const LogLevel level) noexcept
{
    log_min_level.store (static_cast<uint8_t> (level),
                         std::memory_order_relaxed);
}

[[nodiscard]] inline LogLevel
min_log_level () noexcept
{
    return static_cast<LogLevel> (
        log_min_level.load (std::memory_order_relaxed));
}

[[nodiscard]] inline bool
_S_is_enabled (const LogLevel level) noexcept
{
    return static_cast<uint8_t> (level)
           >= log_min_level.load (std::memory_order_relaxed);
}

/* Tri-state ANSI color switch (Rust `Option<bool>` argument of
 * set_ansi_colors). */
enum class AnsiColors : uint8_t
{
    disabled,
    enabled,
    autoDetect,
};

static std::atomic<uint8_t> log_ansi_colors{2}; // 0=disabled, 1=enabled, 2=auto

inline void
set_ansi_colors (const AnsiColors mode) noexcept
{
    log_ansi_colors.store (static_cast<uint8_t> (mode),
                           std::memory_order_relaxed);
}

/* Resolves the mode against the terminal state of the target stream. */
static bool
_S_log_should_use_ansi (const bool stream_is_terminal) noexcept
{
    switch (log_ansi_colors.load (std::memory_order_relaxed))
        {
        case 0:
            return false;
        case 1:
            return true;
        default:
            return stream_is_terminal;
        }
}

/* Error returned by logging operations (Rust `LogError`). */
class LogError
{
  public:
    enum class Kind : uint8_t
    {
        SYSCALL,
        BUFFER_TO_SMALL,
        ENCODING,
        HANDLE_UNAVAILABLE,
        IO,
    };

    constexpr LogError () noexcept = default;

    [[nodiscard]] static constexpr LogError
    syscall_error (const int code) noexcept
    {
        return LogError (Kind::SYSCALL, code);
    }

    [[nodiscard]] static constexpr LogError
    buffer_too_small () noexcept
    {
        return LogError (Kind::BUFFER_TO_SMALL, 0);
    }

    [[nodiscard]] static constexpr LogError
    encoding_error () noexcept
    {
        return LogError (Kind::ENCODING, 0);
    }

    [[nodiscard]] static constexpr LogError
    handle_unavailable () noexcept
    {
        return LogError (Kind::HANDLE_UNAVAILABLE, 0);
    }

    [[nodiscard]] static constexpr LogError
    io_error (const int code) noexcept
    {
        return LogError (Kind::IO, code);
    }

    [[nodiscard]] constexpr Kind
    kind () const noexcept
    {
        return this->_M_kind;
    }

    /* errno for SYSCALL, ErrorKind discriminant for IO, 0 otherwise. */
    [[nodiscard]] constexpr int
    code () const noexcept
    {
        return this->_M_code;
    }

    [[nodiscard]] constexpr bool
    operator== (const LogError &other) const noexcept
    {
        return this->_M_kind == other._M_kind && this->_M_code == other._M_code;
    }

  private:
    Kind _M_kind = Kind::HANDLE_UNAVAILABLE;
    int _M_code = 0;

    constexpr LogError (const Kind kind, const int code) noexcept
        : _M_kind (kind), _M_code (code)
    {
    }
};

static void
log_write_code (FrameSink<64> &sink, const int code) noexcept
{
    long long value = code;
    if (value < 0)
        {
            sink.write ("-");
            value = -value;
        }
    write_uint (sink, static_cast<uint64_t> (value), 0);
}

[[nodiscard]] inline size_t
log_error_message (const LogError &err, char *const buf,
                   const size_t cap) noexcept
{
    if (buf == nullptr || cap == 0)
        return 0;
    FrameSink<64> text;
    switch (err.kind ())
        {
        case LogError::Kind::SYSCALL:
            text.write ("syscall error: ");
            log_write_code (text, err.code ());
            break;
        case LogError::Kind::BUFFER_TO_SMALL:
            text.write ("buffer too small");
            break;
        case LogError::Kind::ENCODING:
            text.write ("encoding error");
            break;
        case LogError::Kind::HANDLE_UNAVAILABLE:
            text.write ("console handle unavailable");
            break;
        case LogError::Kind::IO:
            text.write ("I/O error: ");
            log_write_code (text, err.code ());
            break;
        }
    const SimpleStringView text_view = text.view ();
    const size_t count =
        text_view.size () < cap - 1 ? text_view.size () : cap - 1;
    for (size_t i = 0; i < count; ++i)
        buf[i] = text_view.data ()[i];
    buf[count] = '\0';
    return count;
}

/* Result of a logging call: ok when the message was handed to a writer (or
 * filtered out). */
using LogResult = IoResult<void, LogError>;

/* Converts an io write result into a LogResult (Rust `From<ConsoleError> for
 * LogError`). */
static LogResult
log_from_io (const IoResult<void, ErrorKind> &result) noexcept
{
    if (result.has_value ())
        return LogResult::ok ();
    return LogResult::err (
        LogError::io_error (static_cast<int> (result.error ())));
}

/* Sink for formatted log bytes (Rust `LogWriter` trait). */
class LogWriter
{
  public:
    LogWriter () noexcept = default;
    virtual ~LogWriter () noexcept = default;

    /* Appends bytes to the stdout stream. */
    virtual LogResult
    write_stdout (SimpleStringView bytes) = 0;
    /* Appends bytes to the stderr stream. */
    virtual LogResult
    write_stderr (SimpleStringView bytes) = 0;
    /* Flushes any buffered output. */
    virtual LogResult
    flush () = 0;
};

class DefaultLogWriter final : public LogWriter
{
  public:
    DefaultLogWriter () noexcept = default;

    LogResult
    write_stdout (const SimpleStringView bytes) override
    {
        Stdout out;
        const auto written = out.write_all (to_bytes (bytes));
        if (!written.has_value ())
            return log_from_io (written);
        return log_from_io (out.flush ());
    }

    LogResult
    write_stderr (const SimpleStringView bytes) override
    {
        Stderr out;
        const auto written = out.write_all (to_bytes (bytes));
        if (!written.has_value ())
            return log_from_io (written);
        return log_from_io (out.flush ());
    }

    LogResult
    flush () override
    {
        Stdout out;
        return log_from_io (out.flush ());
    }
};

static DefaultLogWriter _S_log_default_writer{};
static std::atomic<LogWriter *> _S_log_writer{nullptr};

/* Writer set by set_log_writer, or the built-in console writer. */
static LogWriter *
log_get_writer () noexcept
{
    LogWriter *const writer = _S_log_writer.load (std::memory_order_relaxed);
    return writer != nullptr ? writer : &_S_log_default_writer;
}

/* Installs a custom global writer; it must outlive all logging, nullptr
 * restores the default. */
inline void
set_log_writer (LogWriter *const writer) noexcept
{
    _S_log_writer.store (writer, std::memory_order_relaxed);
}

/* The built-in console writer; feed it to set_log_writer to restore default
 * output. */
[[nodiscard]] inline LogWriter *
default_log_writer () noexcept
{
    return &_S_log_default_writer;
}

/* Maximum size of a single log line (Rust LOG_LINE_BUFFER_SIZE). */
inline constexpr size_t _S_log_line_buffer_size = 0x200;

/*
 * Circular (ring) log queue.
 *
 * Producers claim a monotonic ticket with a CAS on _S_log_queue_head,
 * copy the finished line into the claimed slot and publish it with a
 * release store of the commit word (tag in bits 16..17 selects the
 * target stream, the low 16 bits hold the length; 0 means the slot is
 * free or claimed-but-not-yet-committed). The single drain owner pops
 * in ticket order at _S_log_queue_tail with acquire loads, so a slot is
 * only reused after the consumer finished reading it (the producer's
 * acquire load of the tail synchronizes with the consumer's release
 * store). Cursors sit on separate cache lines to avoid false sharing.
 */
/* Number of slots in the circular log queue (always a power of two). */
inline constexpr size_t _S_log_queue_capacity = 0x40;
static_assert ((_S_log_queue_capacity & (_S_log_queue_capacity - 1)) == 0,
               "_S_log_queue_capacity must be a power of two");

static constexpr uint32_t _S_log_slot_tag_stdout = 1u << 16;
static constexpr uint32_t _S_log_slot_tag_stderr = 2u << 16;

static_assert (
    std::atomic<uint32_t>::is_always_lock_free
        && std::atomic<uint64_t>::is_always_lock_free
        && std::atomic<bool>::is_always_lock_free,
    "zzs::base requires lock-free atomics (async-signal-safe logging)");

struct alignas (64) LogSlot
{
    std::atomic<uint32_t> _M_commit{0};
    char _M_data[_S_log_line_buffer_size];
};

alignas (64) static LogSlot _S_log_queue[_S_log_queue_capacity] = {};
alignas (64) static std::atomic<uint64_t> _S_log_queue_head{
    0}; /* next ticket to claim */
alignas (64) static std::atomic<uint64_t> _S_log_queue_tail{
    0}; /* next ticket to consume */
alignas (64) static std::atomic<bool> _S_log_drain_owner{
    false}; /* drains serialized by this token */
static std::atomic<uint64_t> _S_log_overflow_events{
    0}; /* lines written directly because the queue was full */
static std::atomic<uint64_t> _S_log_dropped_events{
    0}; /* lines no writer could accept at all */

/* Upper bound on drain rounds and spin iterations of a single log call. */
static constexpr unsigned _S_log_drain_round_limit = 0x16;
static constexpr unsigned _S_log_drain_spin_limit = 0x1000;

/* Number of lines written directly because the circular queue was full. */
[[nodiscard]] inline uint64_t
_S_log_overflow_count () noexcept
{
    return _S_log_overflow_events.load (std::memory_order_relaxed);
}

/* Number of lines no writer could accept even after the console fallback. */
[[nodiscard]] inline uint64_t
_S_log_dropped_count () noexcept
{
    return _S_log_dropped_events.load (std::memory_order_relaxed);
}

/* Architecture pause between spin iterations (falls back to a compiler fence).
 */
static void
_S_log_relax () noexcept
{
#if (defined(__GNUC__) || defined(__clang__)) \
    && (defined(__i386__) || defined(__x86_64__))
    __builtin_ia32_pause ();
#elif (defined(__GNUC__) || defined(__clang__)) \
    && (defined(__aarch64__) || defined(__arm__))
    __asm__ volatile ("yield" ::: "memory");
#else
    std::atomic_signal_fence (std::memory_order_seq_cst);
#endif
}

/*
 * Writes bytes to their console stream through the installed writer;
 * when that writer fails and it is not already the built-in one, the
 * console writer is used as a fallback. Lines that still cannot be
 * written are counted. Returns true when the bytes were written.
 */
static bool
_S_log_console_write (const bool to_stderr, const SimpleStringView bytes)
{
    LogWriter *const writer = log_get_writer ();
    LogResult result =
        to_stderr ? writer->write_stderr (bytes) : writer->write_stdout (bytes);
    if (!result.has_value () && writer != &_S_log_default_writer)
        result = to_stderr ? _S_log_default_writer.write_stderr (bytes)
                           : _S_log_default_writer.write_stdout (bytes);
    if (!result.has_value ())
        {
            _S_log_dropped_events.fetch_add (1, std::memory_order_relaxed);
            return false;
        }
    return true;
}

/* Claims a ticket, copies the line in and publishes it; false when the queue is
 * full. */
static bool
_S_log_queue_try_push (const char *const data, const size_t len,
                       const bool to_stderr) noexcept
{
    const uint32_t tag =
        to_stderr ? _S_log_slot_tag_stderr : _S_log_slot_tag_stdout;
    uint64_t head = _S_log_queue_head.load (std::memory_order_relaxed);
    for (;;)
        {
            if (head - _S_log_queue_tail.load (std::memory_order_acquire)
                >= _S_log_queue_capacity)
                return false;
            if (_S_log_queue_head.compare_exchange_weak (
                    head, head + 1, std::memory_order_acq_rel,
                    std::memory_order_relaxed))
                break;
            /* CAS reloaded the current head; re-check fullness against it. */
        }
    LogSlot &slot = _S_log_queue[head & (_S_log_queue_capacity - 1)];
    for (size_t i = 0; i < len; ++i)
        slot._M_data[i] = data[i];
    slot._M_commit.store (tag | static_cast<uint32_t> (len),
                          std::memory_order_release);
    return true;
}

/*
 * Drains committed slots in order; must be called while holding the
 * drain token. Stops at the first slot whose producer claimed a ticket
 * but has not committed yet — that producer drains it on its own call.
 */
static void
_S_log_queue_drain ()
{
    for (;;)
        {
            const uint64_t tail =
                _S_log_queue_tail.load (std::memory_order_relaxed);
            if (tail >= _S_log_queue_head.load (std::memory_order_acquire))
                return;
            LogSlot &slot = _S_log_queue[tail & (_S_log_queue_capacity - 1)];
            const uint32_t commit =
                slot._M_commit.load (std::memory_order_acquire);
            if (commit == 0)
                return;
            const size_t len = commit & 0xFFFFu;
            (void) _S_log_console_write ((commit & _S_log_slot_tag_stderr) != 0,
                                         SimpleStringView (slot._M_data, len));
            slot._M_commit.store (0, std::memory_order_relaxed);
            _S_log_queue_tail.store (tail + 1, std::memory_order_release);
        }
}

/* True when the head of the queue already holds a published line. */
static bool
_S_log_queue_has_committed () noexcept
{
    const uint64_t tail = _S_log_queue_tail.load (std::memory_order_acquire);
    if (tail >= _S_log_queue_head.load (std::memory_order_acquire))
        return false;
    return _S_log_queue[tail & (_S_log_queue_capacity - 1)]._M_commit.load (
               std::memory_order_acquire)
           != 0;
}

/*
 * Processes (drains) the circular log queue to the configured writer.
 *
 * Exactly one thread drains at a time (_S_log_drain_owner). A thread that
 * loses the token waits (bounded) for the holder to release it and
 * retries once; the holder re-checks the queue after releasing, so a
 * line committed while it was draining is never stranded. Safe to call
 * at any time, including from a signal handler; a call with an empty
 * queue is a no-op. Every log call runs this after enqueuing.
 */
void
_S_log_process_queue ()
{
    for (unsigned round = 0; round < _S_log_drain_round_limit; ++round)
        {
            bool expected = false;
            if (!_S_log_drain_owner.compare_exchange_strong (
                    expected, true, std::memory_order_acquire,
                    std::memory_order_relaxed))
                {
                    for (unsigned spin = 0; spin < _S_log_drain_spin_limit;
                         ++spin)
                        {
                            if (!_S_log_drain_owner.load (
                                    std::memory_order_acquire))
                                break;
                            _S_log_relax ();
                        }
                    expected = false;
                    if (!_S_log_drain_owner.compare_exchange_strong (
                            expected, true, std::memory_order_acquire,
                            std::memory_order_relaxed))
                        return; /* holder still busy: its post-release re-check
                                   covers us */
                }
            _S_log_queue_drain ();
            _S_log_drain_owner.store (false, std::memory_order_release);
            if (!_S_log_queue_has_committed ())
                return;
        }
}

/*
 * Final emit path: enqueue the finished line, process the queue, and
 * fall back to a direct synchronous write when the queue stays full.
 * Always returns ok — logging never fails.
 */
static void
_S_log_emit (const char *const data, const size_t len, const bool to_stderr)
{
    if (len <= _S_log_line_buffer_size)
        {
            if (_S_log_queue_try_push (data, len, to_stderr))
                {
                    _S_log_process_queue ();
                    return;
                }
            _S_log_process_queue (); /* free slots, then retry once */
            if (_S_log_queue_try_push (data, len, to_stderr))
                {
                    _S_log_process_queue ();
                    return;
                }
            _S_log_overflow_events.fetch_add (1, std::memory_order_relaxed);
        }
    (void) _S_log_console_write (to_stderr, SimpleStringView (data, len));
    return;
}

static size_t
_S_log_copy (char *const buffer, const size_t offset,
             const SimpleStringView text) noexcept
{
    for (size_t i = 0; i < text.size (); ++i)
        buffer[offset + i] = text.data ()[i];
    return text.size ();
}

/* Writes value as exactly width decimal digits (keeps the low digits),
 * advancing idx. */
static void
_S_log_write_fixed (char *const buf, size_t &idx, uint64_t value,
                    const size_t width) noexcept
{
    size_t pos = idx + width;
    for (size_t i = 0; i < width; ++i)
        {
            buf[--pos] = static_cast<char> ('0' + (value % 10));
            value /= 10;
        }
    idx += width;
}

/* Appends "<LEVEL> " (level plus a trailing space) at offset. */
static bool
_S_log_format_level (char *const buffer, const size_t cap, const size_t offset,
                     const LogLevel level, size_t &written) noexcept
{
    const SimpleStringView level_str = as_str (level);
    const size_t required = level_str.size () + 1;
    if (offset + required >= cap)
        return false;
    const size_t copied = _S_log_copy (buffer, offset, level_str);
    buffer[offset + copied] = ' ';
    written = copied + 1;
    return true;
}

/* Appends "message\n", truncating the message to the remaining space minus the
 * newline. */
static bool
_S_log_format_message (char *const buffer, const size_t cap,
                       const size_t offset, const SimpleStringView message,
                       size_t &written) noexcept
{
    if (offset >= cap)
        return false;
    const size_t remaining = cap - offset;
    const size_t msg_len =
        message.size () < remaining - 1 ? message.size () : remaining - 1;
    if (msg_len == 0 && !message.empty ())
        return false;
    const size_t copied =
        _S_log_copy (buffer, offset, message.substr (0, msg_len));
    buffer[offset + copied] = '\n';
    written = copied + 1;
    return true;
}

/*
 * Writes the message as "[<timestamp>]:<LEVEL> <message>\n". ERROR and
 * IRR lines go to stderr, the rest to stdout. The line is built in a
 * fixed 512-byte stack buffer: the timestamp prefix is produced by the
 * manual ISO8601 formatter, long messages are truncated, and ANSI
 * colors around the bracket follow the ansi-colors mode resolved
 * against the target stream's terminal state. The finished line is
 * handed to the circular queue and the queue is processed; the result
 * is always ok — a log call never fails.
 */
inline void
_S_log_with (const LogLevel level, const SimpleStringView message)
{
    if (!_S_is_enabled (level))
        return;
    const bool to_stderr = level >= LogLevel::ERROR;
    char buffer[_S_log_line_buffer_size];
    size_t idx = 0;
    const bool use_color =
        to_stderr ? _S_log_should_use_ansi (Stderr{}.is_terminal ())
                  : _S_log_should_use_ansi (Stdout{}.is_terminal ());
    constexpr SimpleStringView color = "\x1b[32m";
    constexpr SimpleStringView reset_seq = "\x1b[0m";
    const SimpleStringView reset = use_color ? reset_seq : SimpleStringView{};
    if (use_color)
        idx += _S_log_copy (buffer, idx, color);
    size_t part = 0;
    if (_S_log_format_level (buffer, _S_log_line_buffer_size, idx, level, part))
        idx += part;
    /* Reserve room for the trailing newline (and the ANSI reset when colored).
     */
    const size_t reserve = 1 + reset.size ();
    const size_t budget = _S_log_line_buffer_size - idx > reserve
                              ? _S_log_line_buffer_size - idx - reserve
                              : 0;
    size_t written = 0;
    if (_S_log_format_message (buffer, _S_log_line_buffer_size, idx,
                               message.substr (0, budget), written))
        idx += written;
    else if (idx < _S_log_line_buffer_size - 1)
        buffer[idx++] = '\n'; /* pathological: no room left for the message */
    if (!reset.empty () && idx + reset.size () <= _S_log_line_buffer_size)
        idx += _S_log_copy (buffer, idx, reset);
    return _S_log_emit (buffer, idx, to_stderr);
}

/* Writes raw bytes to stdout with no timestamp or level (Rust `log_raw`); never
 * fails. */
inline void
log_raw (const SimpleStringView bytes)
{
    return _S_log_emit (bytes.data (), bytes.size (), false);
}

/*
 * Message style presets: the exact SGR sequences the Rust consoleutil
 * presets build (fg code after the attribute, selective reset).
 */
constexpr SimpleStringView _S_log_style_error_open = "\x1b[1;91m";
constexpr SimpleStringView _S_log_style_error_close = "\x1b[22;39m";
constexpr SimpleStringView _S_log_style_warning_open = "\x1b[1;93m";
constexpr SimpleStringView _S_log_style_warning_close = "\x1b[22;39m";
constexpr SimpleStringView _S_log_style_highlight_open = "\x1b[7m";
constexpr SimpleStringView _S_log_style_highlight_close = "\x1b[27m";
constexpr SimpleStringView _S_log_style_success_open = "\x1b[1;92m";
constexpr SimpleStringView _S_log_style_success_close = "\x1b[22;39m";
constexpr SimpleStringView _S_log_style_info_open = "\x1b[1;94m";
constexpr SimpleStringView _S_log_style_info_close = "\x1b[22;39m";
constexpr SimpleStringView _S_log_style_debug_open = "\x1b[96m";
constexpr SimpleStringView _S_log_style_debug_close = "\x1b[39m";

/* Wraps message in the preset sequences, then logs it at level. */
static void
log (const LogLevel level, const SimpleStringView open,
     const SimpleStringView close, const SimpleStringView message)
{
    FrameSink<512> styled;
    styled.write (open);
    styled.write (message);
    styled.write (close);
    return _S_log_with (level, styled.view ());
}

/* Logs message at ERROR level with the error preset (bright red, bold). */
inline void
log_error (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::ERROR))
        return;
    else
        return log (LogLevel::ERROR, _S_log_style_error_open,
                    _S_log_style_error_close, message);
}

/* Logs message at ERROR level with the warning preset (bright yellow, bold). */
inline void
log_warn (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::ERROR))
        return;
    else
        return log (LogLevel::ERROR, _S_log_style_warning_open,
                    _S_log_style_warning_close, message);
}

/* Logs message at IRR level with the highlight preset (reverse). */
inline void
log_irr (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::IRR))
        return;
    else
        return log (LogLevel::IRR, _S_log_style_highlight_open,
                    _S_log_style_highlight_close, message);
}

/* Logs message at INFO level with the success preset (bright green, bold). */
inline void
log_success (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::INFO))
        return;
    else
        return log (LogLevel::INFO, _S_log_style_success_open,
                    _S_log_style_success_close, message);
}

/* Logs message at INFO level with the info preset (bright blue, bold). */
inline void
log_info (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::INFO))
        return;
    else
        return log (LogLevel::INFO, _S_log_style_info_open,
                    _S_log_style_info_close, message);
}

/* Logs message at DEBUG level with the debug preset (bright cyan). */
inline void
log_debug (const SimpleStringView message)
{
    if (!_S_is_enabled (LogLevel::DEBUG))
        return;
    else
        return log (LogLevel::DEBUG, _S_log_style_debug_open,
                    _S_log_style_debug_close, message);
}

template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
_S_log_with (const LogLevel level, const CharTy (&fmt)[N], const First &first,
             const Rest &...rest)
{
    if (!_S_is_enabled (level))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return _S_log_with (level, SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_error. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_error (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::ERROR))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_error (SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_warn. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_warn (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::ERROR))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_warn (SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_irr. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_irr (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::IRR))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_irr (SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_success. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_success (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::INFO))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_success (SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_info. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_info (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::INFO))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_info (SimpleStringView (message, len));
}

/* Formats the message with stringFormat, then logs it like log_debug. */
template <typename CharTy, unsigned int N, typename First, typename... Rest>
void
log_debug (const CharTy (&fmt)[N], const First &first, const Rest &...rest)
{
    if (!_S_is_enabled (LogLevel::DEBUG))
        return;
    char message[_S_log_line_buffer_size];
    const unsigned int pos = format (message, fmt, first, rest...);
    const size_t len =
        pos < _S_log_line_buffer_size - 1 ? pos : _S_log_line_buffer_size - 1;
    return log_debug (SimpleStringView (message, len));
}
} // namespace zzs::base