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

#include "std/maybe.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>

namespace zzs::base
{
template <typename T>
  constexpr int
  compare (T n1, T n2) noexcept
{
    using traits = std::numeric_limits<int>;
    if (n1 == n2)
        return 0;
    const bool less = n1 < n2;
    const auto diff = static_cast<long long> (less ? n2 - n1 : n1 - n2);
    const auto cap = less ? -static_cast<long long> (traits::min ()) : static_cast<long long> (traits::max ());
    if (diff >= cap)
        return less ? traits::min () : traits::max ();
    return less ? -static_cast<int> (diff) : static_cast<int> (diff);
}
class SvError
{
public:
    enum class Kind
    {
        badAccess,
      };
    SvError () = delete;
    explicit SvError (Kind k) : _M_value (Kind::badAccess) {}
    bool
    operator== (const SvError& other) const
    {
        return _M_value == other._M_value;
    }
    bool
    operator== (const Kind t) const
    {
        return _M_value == t;
    }

private:
    Kind _M_value;
};
constexpr auto
sv_check (const unsigned long size, unsigned long pos)
{
    if (pos > size)
        return Maybe<unsigned long, SvError> (SvError (SvError::Kind::badAccess));
    else
        return Maybe<unsigned long, SvError> (pos);
}

constexpr size_t
sv_limit (const unsigned long size, const unsigned long pos, const unsigned long off) noexcept
{
    const bool testoff = off < size - pos;
    return testoff ? off : size - pos;
}

template <typename CharTy, typename Traits = std::char_traits<std::remove_cv_t<CharTy>>>
class BasicStringView
{
    static_assert (!std::is_array_v<CharTy>);
    static_assert (std::is_trivially_copyable_v<CharTy> && std::is_trivially_default_constructible_v<CharTy>
                   && std::is_standard_layout_v<CharTy>);
    static_assert (std::is_same_v<std::remove_cv_t<CharTy>, typename Traits::char_type>);

public:
    using traits_type = Traits;
    using value_type = CharTy;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using const_iterator = const value_type*;
    using iterator = const_iterator;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    static constexpr size_type not_found = static_cast<size_type> (-1);

    constexpr BasicStringView () noexcept : _M_len{0}, _M_str{nullptr} {}

    constexpr BasicStringView (const BasicStringView&) noexcept = default;
    constexpr BasicStringView (const CharTy* str) noexcept : _M_len{traits_type::length (str)}, _M_str{str} {}
    constexpr BasicStringView (const CharTy* str, const size_type len) noexcept : _M_len{len}, _M_str{str} {}
    template <std::contiguous_iterator Iter, std::sized_sentinel_for<Iter> End>
      requires std::same_as<std::iter_value_t<Iter>, CharTy> && (!std::convertible_to<End, size_type>)
    constexpr BasicStringView (Iter first, End last) noexcept (noexcept (last - first)) :
        _M_len (last - first), _M_str (std::to_address (first))
    {
    }
    template <typename Rg, typename DRg = std::remove_cvref_t<Rg>>
      requires (!std::is_same_v<DRg, BasicStringView>) && std::ranges::contiguous_range<Rg>
                   && std::ranges::sized_range<Rg> && std::is_same_v<std::ranges::range_value_t<Rg>, CharTy>
                   && (!std::is_convertible_v<Rg, const CharTy*>)
                   && (!requires (DRg& d) { d.operator BasicStringView (); })
    constexpr explicit BasicStringView (Rg&& r) noexcept (noexcept (std::ranges::size (r))
                                                          && noexcept (std::ranges::data (r))) :
        _M_len (std::ranges::size (r)), _M_str (std::ranges::data (r))
    {
    }
    BasicStringView (nullptr_t) = delete;
    constexpr BasicStringView&
    operator= (const BasicStringView&) noexcept = default;

    [[nodiscard]]
    constexpr const_iterator
    begin () const noexcept
    {
        return this->_M_str;
    }

    [[nodiscard]]
    constexpr const_iterator
    end () const noexcept
    {
        return this->_M_str + this->_M_len;
    }

    [[nodiscard]]
    constexpr const_iterator
    cbegin () const noexcept
    {
        return this->_M_str;
    }

    [[nodiscard]]
    constexpr const_iterator
    cend () const noexcept
    {
        return this->_M_str + this->_M_len;
    }
    [[nodiscard]]
    constexpr size_type
    size () const noexcept
    {
        return this->_M_len;
    }

    [[nodiscard]]
    constexpr size_type
    max_size () const noexcept
    {
        return (not_found - sizeof (size_type) - sizeof (void*)) / sizeof (value_type) / 4;
    }

    [[nodiscard]]
    constexpr bool
    empty () const noexcept
    {
        return this->_M_len == 0;
    }
    [[nodiscard]]
    constexpr const_reference
    operator[] (size_type pos) const noexcept
    {
        return *(this->_M_str + pos);
    }

    [[nodiscard]]
    constexpr const_reference
    at (size_type pos) const
    {
        return *(this->_M_str + pos);
    }

    [[nodiscard]]
    constexpr const_reference
    front () const noexcept
    {
        return *this->_M_str;
    }

    [[nodiscard]]
    constexpr const_reference
    back () const noexcept
    {
        return *(this->_M_str + this->_M_len - 1);
    }

    [[nodiscard]]
    constexpr const_pointer
    data () const noexcept
    {
        return this->_M_str;
    }

    constexpr void
    remove_prefix (size_type n) noexcept
    {
        this->_M_str += n;
        this->_M_len -= n;
    }

    constexpr void
    remove_suffix (size_type n) noexcept
    {
        this->_M_len -= n;
    }

    constexpr void
    swap (BasicStringView& sv) noexcept
    {
        auto tmp = *this;
        *this = sv;
        sv = tmp;
    }

    constexpr size_type
    copy (CharTy* str, size_type n, size_type pos = 0) const noexcept
    {
        auto pos_checked = sv_check (this->size (), pos);
        if (!pos_checked.has_value ())
            return 0;
        pos = pos_checked.value ();
        const size_type rlen = std::min<size_t> (n, _M_len - pos);
        traits_type::copy (str, data () + pos, rlen);
        return rlen;
    }

    [[nodiscard]]
    constexpr BasicStringView
    substr (size_type pos = 0, const size_type n = not_found) const noexcept
    {
        auto pos_checked = sv_check (this->size (), pos);
        if (!pos_checked.has_value ())
            return BasicStringView ();
        pos = pos_checked.value ();
        const size_type rlen = std::min<size_type> (n, _M_len - pos);
        return BasicStringView{_M_str + pos, rlen};
    }

    [[nodiscard]]
    constexpr int32_t
    compare (BasicStringView str) const noexcept
    {
        const size_type rlen = std::min (this->_M_len, str._M_len);
        int ret = traits_type::compare (this->_M_str, str._M_str, rlen);
        if (ret == 0)
            ret = std::cmp_equal (this->_M_len, str._M_len);
        return ret;
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const size_type pos, const size_type n1, BasicStringView str) const noexcept
    {
        return this->substr (pos, n1).compare (str);
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const size_type pos1, const size_type n1, BasicStringView str, const size_type pos2,
             const size_type n2) const noexcept
    {
        return this->substr (pos1, n1).compare (str.substr (pos2, n2));
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const CharTy* str) const noexcept
    {
        return this->compare (BasicStringView{str});
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const size_type pos, const size_type n1, const CharTy* str) const noexcept
    {
        return this->substr (pos, n1).compare (BasicStringView{str});
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const size_type pos, const size_type n1, const CharTy* str, size_type n2) const noexcept
    {
        return this->substr (pos, n1).compare (BasicStringView (str, n2));
    }
    [[nodiscard]]
    constexpr bool
    starts_with (BasicStringView x) const noexcept
    {
        return _M_len >= x._M_len && traits_type::compare (_M_str, x._M_str, x._M_len) == 0;
    }

    [[nodiscard]]
    constexpr bool
    starts_with (CharTy x) const noexcept
    {
        return !this->empty () && traits_type::eq (this->front (), x);
    }

    [[nodiscard]]
    constexpr bool
    starts_with (const CharTy* x) const noexcept
    {
        return this->starts_with (BasicStringView (x));
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const BasicStringView x) const noexcept
    {
        const auto len = this->size ();
        const auto xlen = x.size ();
        return len >= xlen && traits_type::compare (end () - xlen, x.data (), xlen) == 0;
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const CharTy x) const noexcept
    {
        return !this->empty () && traits_type::eq (this->back (), x);
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const CharTy* x) const noexcept
    {
        return this->ends_with (BasicStringView (x));
    }
    [[nodiscard]]
    constexpr bool
    contains (BasicStringView x) const noexcept
    {
        return this->find (x) != not_found;
    }

    [[nodiscard]]
    constexpr bool
    contains (CharTy x) const noexcept
    {
        for (size_type i = 0; i < this->_M_len; ++i)
            {
                if (this->_M_str[i] == x)
                    return true;
            }
        return false;
    }

    [[nodiscard]]
    constexpr bool
    contains (const CharTy* x) const noexcept
    {
        return this->find (x) != not_found;
    }

    [[nodiscard]]
    constexpr size_type
    find (BasicStringView str, const size_type pos = 0) const noexcept
    {
        return this->find (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    find (const CharTy* str, size_type pos = 0) const noexcept
    {
        const size_type len = traits_type::length (str);
        if (pos > this->_M_len)
            return not_found;
        if (len == 0)
            return pos;
        const CharTy elem0 = str[0];
        const CharTy* const data = this->_M_str;
        const CharTy* first = data + pos;
        const CharTy* const last = data + this->_M_len;
        size_type remaining = this->_M_len - pos;

        while (remaining >= len)
            {
                first = traits_type::find (first, remaining - len + 1, elem0);
                if (!first)
                    return not_found;
                if (traits_type::compare (first, str, len) == 0)
                    return first - data;
                remaining = last - ++first;
            }
        return not_found;
    }

    [[nodiscard]]
    constexpr size_type
    rfind (BasicStringView str, size_type pos = not_found) const noexcept
    {
        return this->rfind (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    rfind (const CharTy* str, size_type pos = not_found) const noexcept
    {
        return this->rfind (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (BasicStringView str, size_type pos = 0) const noexcept
    {
        return this->find_first_of (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (CharTy c, size_type pos = 0) const noexcept
    {
        return this->find (c, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (const CharTy* str, size_type pos = 0) const noexcept
    {
        return this->find_first_of (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (BasicStringView str, size_type pos = not_found) const noexcept
    {
        return this->find_last_of (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (CharTy c, size_type pos = not_found) const noexcept
    {
        return this->rfind (c, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (const CharTy* str, size_type pos = not_found) const noexcept
    {
        return this->find_last_of (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_not_of (BasicStringView str, size_type pos = 0) const noexcept
    {
        return this->find_first_not_of (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_not_of (const CharTy* str, size_type pos = 0) const noexcept
    {
        return this->find_first_not_of (str, pos, traits_type::length (str));
    }

    [[nodiscard]]
    constexpr size_type
    find_last_not_of (BasicStringView str, size_type pos = not_found) const noexcept
    {
        return this->find_last_not_of (str._M_str, pos, str._M_len);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_not_of (const CharTy* str, size_type pos = not_found) const noexcept
    {
        return this->find_last_not_of (str, pos, traits_type::length (str));
    }

private:
    size_type _M_len;
    const CharTy* _M_str;
};

template <unsigned int N, typename CharTy, typename Traits = std::char_traits<std::remove_cv_t<CharTy>>>
struct BasicTemplatedStringView
{
    static_assert (!std::is_array_v<CharTy>);
    static_assert (std::is_trivially_copyable_v<CharTy> && std::is_trivially_default_constructible_v<CharTy>
                   && std::is_standard_layout_v<CharTy>);
    static_assert (std::is_same_v<std::remove_cv_t<CharTy>, typename Traits::char_type>);

    using traits_type = Traits;
    using value_type = CharTy;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using const_iterator = const value_type*;
    using iterator = const_iterator;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static constexpr size_type nfound = static_cast<size_type> (-1);
    static constexpr size_type capacity_value = N;

    constexpr BasicTemplatedStringView () noexcept = default;
    constexpr BasicTemplatedStringView (const BasicTemplatedStringView&) noexcept = default;
    constexpr BasicTemplatedStringView (const CharTy (&str)[N]) noexcept :
        BasicTemplatedStringView (str, std::make_index_sequence<N>{})
    {
    }
    template <size_t... Ts>
    constexpr BasicTemplatedStringView (const CharTy (&str)[N], std::index_sequence<Ts...>) noexcept :
        _M_str{str[Ts]...}
    {
    }
    template <typename Rg, typename DRg = std::remove_cvref_t<Rg>>
      requires (!std::is_same_v<DRg, BasicTemplatedStringView>)
               && std::ranges::contiguous_range<Rg> && std::ranges::sized_range<Rg>
               && std::is_same_v<std::ranges::range_value_t<Rg>, CharTy> && (!std::is_convertible_v<Rg, const CharTy*>)
               && (!requires (DRg& d) { d.operator BasicTemplatedStringView (); })
    constexpr explicit BasicTemplatedStringView (Rg&& r) noexcept (noexcept (std::ranges::size (r))
                                                                   && noexcept (std::ranges::data (r))) :
        _M_str (std::ranges::data (r))
    {
    }
    BasicTemplatedStringView (nullptr_t) = delete;
    constexpr BasicTemplatedStringView&
    operator= (const BasicTemplatedStringView&) noexcept = default;

    [[nodiscard]]
    constexpr const_iterator
    begin () const noexcept
    {
        return this->_M_str;
    }

    [[nodiscard]]
    constexpr const_iterator
    end () const noexcept
    {
        return this->_M_str + N;
    }

    [[nodiscard]]
    constexpr const_iterator
    cbegin () const noexcept
    {
        return this->_M_str;
    }

    [[nodiscard]]
    constexpr const_iterator
    cend () const noexcept
    {
        return this->_M_str + N;
    }
    [[nodiscard]]
    constexpr size_type
    size () const noexcept
    {
        return N;
    }
    [[nodiscard]]
    constexpr
    operator BasicStringView<CharTy> () const noexcept
    {
        return BasicStringView<CharTy> (this->_M_str, N);
    }

    [[nodiscard]]
    constexpr size_type
    max_size () const noexcept
    {
        return (nfound - sizeof (size_type) - sizeof (void*)) / sizeof (value_type) / 4;
    }
    [[nodiscard]]
    constexpr const_reference
    operator[] (size_type pos) const noexcept
    {
        return *(this->_M_str + pos);
    }

    [[nodiscard]]
    constexpr const_reference
    at (size_type pos) const
    {
        return *(this->_M_str + pos);
    }

    [[nodiscard]]
    constexpr const_reference
    front () const noexcept
    {
        return *this->_M_str;
    }

    [[nodiscard]]
    constexpr const_reference
    back () const noexcept
    {
        return *(this->_M_str + N - 1);
    }

    [[nodiscard]]
    constexpr const_pointer
    data () const noexcept
    {
        return this->_M_str;
    }

    constexpr void
    swap (BasicTemplatedStringView& sv) noexcept
    {
        auto tmp = *this;
        *this = sv;
        sv = tmp;
    }

    constexpr size_type
    copy (CharTy* str, size_type n, size_type pos = 0) const noexcept
    {
        auto pos_checked = sv_check (this->size (), pos);
        if (!pos_checked.has_value ())
            return 0;
        pos = pos_checked.value ();
        const size_type rlen = std::min<size_t> (n, N - pos);
        traits_type::copy (str, data () + pos, rlen);
        return rlen;
    }

    [[nodiscard]]
    constexpr int32_t
    compare (const BasicTemplatedStringView str) const noexcept
    {
        const size_type rlen = std::min (this->capacity_value, str.capacity_value);
        int ret = traits_type::compare (this->_M_str, str._M_str, rlen);
        if (ret == 0)
            ret = ::zzs::base::compare (this->size (), str.size ());
        return ret;
    }
    [[nodiscard]]
    constexpr bool
    starts_with (BasicTemplatedStringView x) const noexcept
    {
        return this->size () >= x.size () && traits_type::compare (_M_str, x._M_str, x.size ()) == 0;
    }

    [[nodiscard]]
    constexpr bool
    starts_with (CharTy x) const noexcept
    {
        return N > 0 && traits_type::eq (this->front (), x);
    }

    [[nodiscard]]
    constexpr bool
    starts_with (const CharTy* x) const noexcept
    {
        return this->starts_with (BasicStringView (x));
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const BasicTemplatedStringView x) const noexcept
    {
        const auto len = this->size ();
        const auto xlen = x.size ();
        return len >= xlen && traits_type::compare (end () - xlen, x.data (), xlen) == 0;
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const CharTy x) const noexcept
    {
        return N > 0 && traits_type::eq (this->back (), x);
    }

    [[nodiscard]]
    constexpr bool
    ends_with (const CharTy* x) const noexcept
    {
        return this->ends_with (BasicStringView (x));
    }
    [[nodiscard]]
    constexpr bool
    contains (BasicTemplatedStringView x) const noexcept
    {
        return this->find (x) != nfound;
    }

    [[nodiscard]]
    constexpr bool
    contains (CharTy x) const noexcept
    {
        for (size_type i = 0; i < N; ++i)
            {
                if (this->_M_str[i] == x)
                    return true;
            }
        return false;
    }

    [[nodiscard]]
    constexpr bool
    contains (const CharTy* x) const noexcept
    {
        return this->find (x) != nfound;
    }

    [[nodiscard]]
    constexpr size_type
    find (BasicTemplatedStringView str, const size_type pos = 0) const noexcept
    {
        return this->find (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    find (const CharTy* str, size_type pos = 0) const noexcept
    {
        const size_type len = traits_type::length (str);
        if (pos > N)
            return nfound;
        if (len == 0)
            return pos;
        const CharTy elem0 = str[0];
        const CharTy* const data = this->_M_str;
        const CharTy* first = data + pos;
        const CharTy* const last = data + N;
        size_type remaining = N - pos;
        while (remaining >= len)
            {
                first = traits_type::find (first, remaining - len + 1, elem0);
                if (!first)
                    return nfound;
                if (traits_type::compare (first, str, len) == 0)
                    return first - data;
                remaining = last - ++first;
            }
        return nfound;
    }

    [[nodiscard]]
    constexpr size_type
    rfind (BasicTemplatedStringView str, size_type pos = nfound) const noexcept
    {
        return this->rfind (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    rfind (const CharTy* str, size_type pos = nfound) const noexcept
    {
        return this->rfind (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (BasicTemplatedStringView str, size_type pos = 0) const noexcept
    {
        return this->find_first_of (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (CharTy c, size_type pos = 0) const noexcept
    {
        return this->find (c, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_of (const CharTy* str, size_type pos = 0) const noexcept
    {
        return this->find_first_of (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (BasicTemplatedStringView str, size_type pos = nfound) const noexcept
    {
        return this->find_last_of (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (CharTy c, size_type pos = nfound) const noexcept
    {
        return this->rfind (c, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_of (const CharTy* str, size_type pos = nfound) const noexcept
    {
        return this->find_last_of (str, pos);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_not_of (BasicTemplatedStringView str, size_type pos = 0) const noexcept
    {
        return this->find_first_not_of (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    find_first_not_of (const CharTy* str, size_type pos = 0) const noexcept
    {
        return this->find_first_not_of (str, pos, traits_type::length (str));
    }

    [[nodiscard]]
    constexpr size_type
    find_last_not_of (BasicTemplatedStringView str, size_type pos = nfound) const noexcept
    {
        return this->find_last_not_of (str._M_str, pos, N);
    }

    [[nodiscard]]
    constexpr size_type
    find_last_not_of (const CharTy* str, size_type pos = nfound) const noexcept
    {
        return this->find_last_not_of (str, pos, traits_type::length (str));
    }
    value_type _M_str[N]{};
};

template <typename T>
struct IsStruturalStringView : std::false_type
{
};
template <unsigned int N, typename CharTy, typename Traits>
struct IsStruturalStringView<BasicTemplatedStringView<N, CharTy, Traits>> : std::true_type
{
};

template <typename, typename>
struct IsStringConvertible : std::false_type
{
};
template <typename T>
struct IsStringConvertible<T, char> : std::bool_constant<std::is_convertible_v<T, BasicStringView<char>>>
{
};
template <typename T>
struct IsStringConvertible<T, char8_t> : std::bool_constant<std::is_convertible_v<T, BasicStringView<char>>>
{
};
template <typename T>
struct IsStringConvertible<T, char16_t> : std::bool_constant<std::is_convertible_v<T, BasicStringView<char>>>
{
};
template <typename, typename>
struct IsStringConstructible : std::false_type
{
};
template <typename T>
struct IsStringConstructible<T, char> : std::bool_constant<std::is_constructible_v<BasicStringView<char>, T>>
{
};
template <typename, typename>
struct IsNothrowStringConvertible : std::false_type
{
};
template <typename T>
struct IsNothrowStringConvertible<T, char>
    : std::bool_constant<std::is_nothrow_convertible_v<T, BasicStringView<char>>>
{
};
template <typename T>
struct IsNothrowStringConvertible<T, char8_t>
    : std::bool_constant<std::is_nothrow_convertible_v<T, BasicStringView<char8_t>>>
{
};
template <typename T>
struct IsNothrowStringConvertible<T, char16_t>
    : std::bool_constant<std::is_nothrow_convertible_v<T, BasicStringView<char16_t>>>
{
};
template <typename, typename>
struct IsNothrowStringConstructible : std::false_type
{
};
template <typename T>
struct IsNothrowStringConstructible<T, char> : std::bool_constant<std::is_nothrow_constructible_v<T, char>>
{
};
template <typename T>
struct IsNothrowStringConstructible<T, char8_t> : std::bool_constant<std::is_nothrow_constructible_v<T, char8_t>>
{
};
template <typename T>
struct IsNothrowStringConstructible<T, char16_t> : std::bool_constant<std::is_nothrow_constructible_v<T, char16_t>>
{
};
template <typename T, typename CharTy>
using IsStringNothrowConvertible = IsNothrowStringConvertible<T, CharTy>;
template <typename T, typename CharTy>
using IsStringNothrowConstructible = IsNothrowStringConstructible<T, CharTy>;

template <unsigned int N1, unsigned int N2, typename CharTy, typename Traits>
[[nodiscard]]
constexpr auto
operator+ (const BasicTemplatedStringView<N1, CharTy, Traits>& lhs,
           const BasicTemplatedStringView<N2, CharTy, Traits>& rhs) noexcept
{
    BasicTemplatedStringView<N1 + N2, CharTy, Traits> result;
    for (size_t i = 0; i < N1; ++i)
        {
            result._M_str[i] = lhs[i];
        }
    for (size_t i = 0; i < N2; ++i)
        {
            result._M_str[N1 + i] = rhs[i];
        }
    return result;
}

template <typename CharTraits>
constexpr auto
char_traits_cmp (const int cmp) noexcept
{
    if constexpr (requires { typename CharTraits::comparison_category; })
        {
            using cat = CharTraits::comparison_category;
            static_assert (!std::is_void_v<std::common_comparison_category_t<cat>>);
            return static_cast<cat> (cmp <=> 0);
        }
    else
        return static_cast<std::weak_ordering> (cmp <=> 0);
}

template <std::contiguous_iterator Iter, std::sized_sentinel_for<Iter> End>
BasicStringView (Iter, End) -> BasicStringView<std::iter_value_t<Iter>>;

template <std::ranges::contiguous_range Rg>
BasicStringView (Rg&&) -> BasicStringView<std::ranges::range_value_t<Rg>>;

template <typename CharTy, typename Traits>
[[nodiscard]]
constexpr bool
operator== (BasicStringView<CharTy, Traits> x, std::type_identity_t<BasicStringView<CharTy, Traits>> y) noexcept
{
    return x.size () == y.size () && x.compare (y) == 0;
}

template <typename CharTy, typename Traits>
[[nodiscard]]
constexpr auto
operator<=> (BasicStringView<CharTy, Traits> x, std::type_identity_t<BasicStringView<CharTy, Traits>> y) noexcept
    -> decltype (char_traits_cmp<Traits> (0))
{
    return char_traits_cmp<Traits> (x.compare (y));
}
using SimpleStringView = BasicStringView<char>;
using U8StringView = BasicStringView<char8_t>;
using U16StringView = BasicStringView<char16_t>;
template <unsigned int N>
using SimpleTemplatedStringView = BasicTemplatedStringView<N, char>;
template <unsigned int N>
using U8TemplatedStringView = BasicTemplatedStringView<N, char8_t>;
template <unsigned int N>
using U16TemplatedStringView = BasicTemplatedStringView<N, char16_t>;
}