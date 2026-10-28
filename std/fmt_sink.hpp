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
template <unsigned int N> class FrameSink
{
private:
    static_assert (N > 0);
    char _M_buf[N]{};
    size_t _M_len = 0;

public:
    constexpr FrameSink () noexcept = default;

    [[nodiscard]] constexpr size_t
    size () const noexcept
    {
        return this->_M_len;
    }
    [[nodiscard]] constexpr bool
    empty () const noexcept
    {
        return this->_M_len == 0;
    }
    [[nodiscard]] constexpr const char *
    data () const noexcept
    {
        return this->_M_buf;
    }
    [[nodiscard]] constexpr SimpleStringView
    view () const noexcept
    {
        return SimpleStringView (this->_M_buf, this->_M_len);
    }

    constexpr size_t
    write (const SimpleStringView str) noexcept
    {
        const size_t avail = N - this->_M_len;
        const size_t take = str.size () < avail ? str.size () : avail;
        for (size_t i = 0; i < take; ++i)
            this->_M_buf[this->_M_len + i] = str.data ()[i];
        this->_M_len += take;
        return take;
    }
    constexpr size_t
    write (const char c) noexcept
    {
        const size_t avail = N - this->_M_len;
        const size_t take = 1 < avail ? 1 : avail;
        if (take > 0)
            {
                this->_M_buf[this->_M_len++] = c;
                return take;
            }
        else
            {
                return take;
            }
    }

    /* Resets the sink to empty so its buffer can be reused for the next line.
     */
    constexpr void
    clear () noexcept
    {
        this->_M_len = 0;
    }
};


/* Writes value as decimal, left-padded with spaces to width columns. */
template <typename Sink>
constexpr void
write_uint (Sink &sink, uint64_t value, const size_t width) noexcept
{
    char digits[20];
    size_t count = 0;
    do
    {
        digits[count++] = static_cast<char> ('0' + (value % 10));
        value /= 10;
    }
    while (value != 0);
    char buf[24];
    size_t pos = 0;
    const size_t spaces = width > count ? width - count : 0;
    for (size_t i = 0; i < spaces && pos < sizeof (buf); ++i)
        buf[pos++] = ' ';
    while (count > 0 && pos < sizeof (buf))
        buf[pos++] = digits[--count];
    sink.write (SimpleStringView (buf, pos));
}
inline constexpr char _S_prettify_hex_digits[] = "0123456789abcdef";

/*
 * Writes value as lowercase hex with at least min_digits digits
 * (zero-padded on the left), without a "0x" prefix.
 */
template <typename Sink>
constexpr void
write_hex (Sink &sink, const uint64_t value, const size_t min_digits) noexcept
{
    char digits[16];
    size_t count = 0;
    uint64_t v = value;
    do
    {
        digits[count++] = _S_prettify_hex_digits[v & 0xF];
        v >>= 4;
    }
    while (v != 0);
    char buf[16];
    size_t pos = 0;
    size_t pad = count > min_digits ? count : min_digits;
    if (pad > sizeof (buf))
    {
        pad = sizeof (buf);
    }
    for (size_t i = count; i < pad; ++i)
    {
        buf[pos++] = '0';
    }
    while (count > 0)
    {
        buf[pos++] = digits[--count];
    }
    sink.write (SimpleStringView (buf, pos));
}

}