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

#include "kernel/kernel_config.hpp"
#include "std/maybe.hpp"
#include "std/string_view.hpp"

#include <cstdint>

/**
 * @brief Result codes for the preprocessing bank.
 *
 * Errors are values, as everywhere in this product: the kernel never throws and
 * never logs.  Hot paths -- one decoded sample, one resonator step -- return
 * `Status` directly because a `Maybe` would cost the storage of a payload they
 * do not have; only the boundaries that actually produce something (a decoded
 * packet, a finished feature frame) return `zzs::base::Maybe<Value, Error>`,
 * which is the same split mllee uses in `Core::run`.
 *
 * The vocabulary is deliberately wider than mllee's: the failure modes here are
 * a link that truncates, a stream that desynchronizes and a clock that drifts,
 * none of which a model reader can express.
 */
namespace zzs::kernel {
struct Error {
  enum class Kind : uint8_t {
    OK = 0,
    /* Framing and link. */
    TRUNCATED,      /**< The source ran out before the packet did. */
    BAD_MAGIC,      /**< The start byte is not the module's own. */
    BAD_VERSION,    /**< The header version is not one this build decodes. */
    BAD_CRC,        /**< The payload does not match its trailing checksum. */
    BAD_TYPE,       /**< A declared packet type this build cannot decode. */
    BAD_LENGTH,     /**< A length field that contradicts the type. */
    BAD_SEQUENCE,   /**< A retransmit or a gap the dedup cannot accept. */
    STUFF_OVERFLOW, /**< Escaped bytes exceeded the frame budget. */
    QUEUE_FULL,     /**< The receive queue wrapped before the reader drained. */
    /* Sample path. */
    OUT_OF_RANGE,  /**< A value or an index outside a documented bound. */
    STALE,         /**< The channel has not produced since its timeout. */
    NOT_READY,     /**< A window asked for before it was filled. */
    GRID_UNSTABLE, /**< The clock moved too far for the grid to hold. */
    /* Module limits. */
    UNSUPPORTED,     /**< A rate, axis count or op the build has no path for. */
    BUDGET_EXCEEDED, /**< A caller buffer smaller than the documented need. */
    INTERNAL,        /**< An invariant the stage table says cannot happen. */
  };

  Error () = delete;
  explicit constexpr Error (const Kind kind) noexcept : _M_kind (kind) {
  }

  [[nodiscard]] constexpr Kind
  kind () const noexcept {
    return this->_M_kind;
  }

  [[nodiscard]] constexpr bool
  is_ok () const noexcept {
    return this->_M_kind == Kind::OK;
  }

  [[nodiscard]] constexpr bool
  operator== (const Error &other) const noexcept {
    return this->_M_kind == other._M_kind;
  }

  [[nodiscard]] constexpr bool
  operator!= (const Error &other) const noexcept {
    return this->_M_kind != other._M_kind;
  }

  /** @brief True for every kind except OK, so a call site can read `if (err)`. */
  [[nodiscard]] constexpr explicit
  operator bool () const noexcept {
    return this->_M_kind != Kind::OK;
  }

private:
  Kind _M_kind;
};

using Status = Error::Kind;

[[nodiscard]] constexpr bool
status_ok (const Status status) noexcept {
  return status == Status::OK;
}

/**
 * @brief The code's name, for the host-side test and telemetry boundary.
 *
 * Returned as a `SimpleStringView` rather than a `const char *` because every
 * caller either compares it or formats it, and both want a length; the literals
 * are template arguments, so the view's size is part of its type and nothing is
 * measured at run time.
 */
[[nodiscard]] constexpr zzs::base::SimpleStringView
status_name (const Status status) noexcept {
  switch (status) {
  case Status::OK:
    return "ok";
  case Status::TRUNCATED:
    return "truncated";
  case Status::BAD_MAGIC:
    return "bad_magic";
  case Status::BAD_VERSION:
    return "bad_version";
  case Status::BAD_CRC:
    return "bad_crc";
  case Status::BAD_TYPE:
    return "bad_type";
  case Status::BAD_LENGTH:
    return "bad_length";
  case Status::BAD_SEQUENCE:
    return "bad_sequence";
  case Status::STUFF_OVERFLOW:
    return "stuff_overflow";
  case Status::QUEUE_FULL:
    return "queue_full";
  case Status::OUT_OF_RANGE:
    return "out_of_range";
  case Status::STALE:
    return "stale";
  case Status::NOT_READY:
    return "not_ready";
  case Status::GRID_UNSTABLE:
    return "grid_unstable";
  case Status::UNSUPPORTED:
    return "unsupported";
  case Status::BUDGET_EXCEEDED:
    return "budget_exceeded";
  case Status::INTERNAL:
    return "internal";
  }
  return "unknown";
}

/** @brief Public boundary result type: a produced value or one kernel error. */
template <typename Value> using Result = zzs::base::Maybe<Value, Error>;

/**
 * @brief Wrap a `Status` in the error side of a Result.
 *
 * The one-line form exists so a stage's early returns read the same as mllee's
 * `Core::_M_fail`, and so a caller never has to remember whether `Error`'s
 * constructor is explicit.
 */
[[nodiscard]] constexpr Result<void>
result_error (const Status status) noexcept {
  return Result<void> (Error (status));
}
} // namespace zzs::kernel
