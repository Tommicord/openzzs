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

#include <cstddef>
#include <type_traits>
#include <utility>

constexpr struct _S_SynthImpl
{
    template <typename _T> using boolean_testable = bool;
    template <typename _T, typename _ValUp>
    [[nodiscard]]
    constexpr auto
    operator() (const _T &t, const _ValUp &u) const noexcept
        requires requires {
        { t < u } -> std::convertible_to<bool>;
        { u < t } -> std::convertible_to<bool>;
    }
{
    if constexpr (std::three_way_comparable_with<_T, _ValUp>)
        return t <=> u;
    else
        {
            if (t < u)
                return std::weak_ordering::less;
            else if (u < t)
                return std::weak_ordering::greater;
            else
                return std::weak_ordering::equivalent;
        }
}
} _S_synth_three = {};
template <typename _T, typename ValUp = _T>
using SynthType =
    decltype (_S_synth_three (std::declval<_T &> (), std::declval<ValUp &> ()));

namespace zzs::base
{
template <typename T> struct BasicIterator
{
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using difference_type = ptrdiff_t;
    using size_type = size_t;

    constexpr BasicIterator () noexcept = default;
    [[nodiscard]] explicit constexpr
    operator bool () const noexcept
    {
        return this->get () != nullptr;
    }

    explicit constexpr BasicIterator (const pointer iter) noexcept
        : _M_current (iter)
    {
    }
    explicit constexpr BasicIterator (const const_pointer iter) noexcept
        : _M_current (iter)
    {
    }
    explicit constexpr BasicIterator (reference iter) noexcept
        : _M_current (iter)
    {
    }
    explicit constexpr BasicIterator (const_reference iter) noexcept
        : _M_current (iter)
    {
    }

    template <typename _T>
        requires std::is_pointer_v<_T>
                 && std::is_same_v<std::remove_pointer_t<_T>, _T>
    explicit constexpr BasicIterator (_T iter) noexcept : _M_current (iter)
    {
    }

    constexpr BasicIterator &
    operator= (const BasicIterator &other) noexcept
    {
        this->_M_current = other._M_current;
        return *this;
    }

    template <typename Iterer>
        requires std::is_convertible_v<Iterer, T>
    explicit constexpr BasicIterator (const value_type &iter) noexcept
        : _M_current (iter.get ())
    {
    }
    [[nodiscard]] constexpr bool
    operator== (const BasicIterator &other) const noexcept
    {
        return this->get () == other.get ();
    }
    [[nodiscard]] constexpr bool
    operator!= (nullptr_t) const noexcept
    {
        return this->get () != nullptr;
    }
    [[nodiscard]] friend constexpr bool
    operator== (nullptr_t, const BasicIterator &other) noexcept
    {
        return other.get () == nullptr;
    }
    [[nodiscard]] friend constexpr bool
    operator!= (nullptr_t, const BasicIterator &other) noexcept
    {
        return other.get () != nullptr;
    }
    [[nodiscard]] constexpr bool
    operator!= (const BasicIterator &other) const noexcept
    {
        return this->get () != other.get ();
    }
    [[nodiscard]] friend constexpr bool
    operator== (const BasicIterator &lhs, const BasicIterator &rhs) noexcept
    {
        return lhs.get () == rhs.get ();
    }
    [[nodiscard]] friend constexpr bool
    operator!= (const BasicIterator &lhs, const BasicIterator &rhs) noexcept
    {
        return lhs.get () != rhs.get ();
    }
    [[nodiscard]] constexpr const_pointer
    get () const noexcept
    {
        return this->_M_current;
    }
    [[nodiscard]] constexpr const_reference
    operator* () const noexcept
    {
        return *this->get ();
    }
    [[nodiscard]] constexpr const_pointer
    operator->() const noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr reference
    operator[] (difference_type n) noexcept
    {
        return *(this->_M_current + n);
    }
    [[nodiscard]] constexpr const_reference
    operator[] (difference_type n) const noexcept
    {
        return *(this->_M_current + n);
    }
    constexpr
    operator pointer () noexcept
    {
        return this->get ();
    }
    constexpr
    operator const_pointer () const noexcept
    {
        return this->get ();
    }

    constexpr BasicIterator &
    operator++ () noexcept
    {
        ++this->_M_current;
        return *this;
    }
    constexpr BasicIterator &
    operator++ (int) noexcept
    {
        return BasicIterator (this->_M_current++);
    }
    constexpr BasicIterator &
    operator-- () noexcept
    {
        --this->_M_current;
        return *this;
    }

    constexpr BasicIterator
    operator-- (int) noexcept
    {
        return BasicIterator (this->_M_current--);
    }
    constexpr BasicIterator &
    operator+= (difference_type n) noexcept
    {
        this->_M_current += n;
        return *this;
    }
    [[nodiscard]] constexpr BasicIterator
    operator+ (difference_type n) const noexcept
    {
        return BasicIterator (this->_M_current + n);
    }
    [[nodiscard]] constexpr difference_type
    operator+ (const BasicIterator &other) const noexcept
    {
        return static_cast<difference_type> (this->get () + other.get ());
    }
    constexpr BasicIterator &
    operator-= (difference_type n) noexcept
    {
        this->_M_current -= n;
        return *this;
    }
    [[nodiscard]] constexpr BasicIterator
    operator- (difference_type n) const noexcept
    {
        return BasicIterator (this->_M_current - n);
    }
    [[nodiscard]] constexpr difference_type
    operator- (const BasicIterator &other) const noexcept
    {
        return static_cast<difference_type> (this->get () - other.get ());
    }
    template <typename Iterer>
    constexpr bool
    operator== (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (this->get () == other.get ()))
        requires requires {
            { this->get () == other.get () } -> std::convertible_to<bool>;
        }
    {
        return this->get () == other.get ();
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr SynthType<T, Iterer>
    operator<=> (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (_S_synth_three (this->get (), other.get ())))
    {
        return __synth (this->get (), other.get ());
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr SynthType<Iterer>
    operator<=> (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (_S_synth_three (this->get (), other.get ())))
    {
        return _S_synth_three (this->get (), other.get ());
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr auto
    operator- (const BasicIterator<Iterer> &rhs) const noexcept
        -> decltype (this->get () - rhs.get ())
    {
        return this->get () - rhs.get ();
    }

    template <typename Iterer>
    [[nodiscard]] constexpr auto
    operator+ (difference_type n) const noexcept -> decltype (this->get () + n)
    {
        return BasicIterator<Iterer> (this->get () + n);
    }
    template <typename Iterer>
    [[nodiscard]] constexpr auto
    operator+ (const BasicIterator<Iterer> &other) const noexcept
        -> decltype (this->get () + other.get ())
    {
        return BasicIterator<Iterer> (this->get () + other.get ());
    }

  private:
    const_pointer _M_current{};
};
template <typename T> class BorrowedIterator;

template <typename T> struct IteratorDeleter
{
    constexpr IteratorDeleter () noexcept = default;

    template <typename ValUp>
        requires (std::is_convertible_v<ValUp *, T *>)
    explicit constexpr IteratorDeleter (const IteratorDeleter<ValUp> &) noexcept
    {
    }
    constexpr void
    operator() (T *ptr) const
    {
        static_assert (!std::is_void_v<T>,
                       "zzs::base::IteratorDeleter can't delete pointer "
                       "to incomplete type");
        static_assert (sizeof (T) > 0, "zzs::base::IteratorDeleter can't "
                                       "delete pointer to incomplete type");
        delete ptr;
    }
};

template <typename T, typename Deleter = IteratorDeleter<T>> class OwnedIterator
{
  public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using deleter_type = Deleter;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    constexpr OwnedIterator () noexcept = default;
    explicit constexpr OwnedIterator (pointer value) noexcept
        : _M_current (value)
    {
    }
    OwnedIterator (const OwnedIterator &) = delete;
    OwnedIterator &
    operator= (const OwnedIterator &) = delete;
    constexpr OwnedIterator (OwnedIterator &&other) noexcept
        : _M_current (std::exchange (other._M_current, pointer{}))
    {
    }
    constexpr OwnedIterator &
    operator= (OwnedIterator &&other) noexcept = delete;
    constexpr ~OwnedIterator ()
    {
        this->reset ();
    }
    [[nodiscard]] constexpr explicit
    operator bool () const noexcept
    {
        return this->get () != nullptr;
    }
    [[nodiscard]] constexpr
    operator pointer () noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr
    operator const_pointer () const noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr const_pointer
    get () const noexcept
    {
        return this->_M_current;
    }
    [[nodiscard]] constexpr const_reference
    operator* () const noexcept
    {
        return *this->get ();
    }
    [[nodiscard]] constexpr const_pointer
    operator->() const noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr reference
    operator[] (difference_type n) noexcept
    {
        return *(this->_M_current + n);
        ;
    }
    [[nodiscard]] constexpr const_reference
    operator[] (difference_type n) const noexcept
    {
        return *(this->_M_current + n);
    }
    constexpr pointer
    release () noexcept
    {
        return static_cast<pointer> (
            std::exchange (this->_M_current, pointer{}));
    }
    constexpr void
    reset (pointer value = nullptr) noexcept
    {
        if (this->get () != value)
            {
                if (this->get () != nullptr)
                    Deleter{}(this->get ());
                this->_M_current = value;
            }
    }
    [[nodiscard]] constexpr bool
    operator== (const OwnedIterator &other) const noexcept
    {
        return this->get () == other.get ();
    }
    [[nodiscard]] constexpr bool
    operator== (nullptr_t) const noexcept
    {
        return this->get () == nullptr;
    }
    [[nodiscard]] constexpr bool
    operator!= (nullptr_t) const noexcept
    {
        return this->get () != nullptr;
    }
    [[nodiscard]] friend constexpr bool
    operator== (nullptr_t, const OwnedIterator &other) noexcept
    {
        return other == nullptr;
    }
    [[nodiscard]] friend constexpr bool
    operator!= (nullptr_t, const OwnedIterator &other) noexcept
    {
        return other != nullptr;
    }
    [[nodiscard]] constexpr bool
    operator!= (const OwnedIterator &other) const noexcept
    {
        return this->get () != other.get ();
    }
    [[nodiscard]] friend constexpr bool
    operator== (const OwnedIterator &lhs, const OwnedIterator &rhs) noexcept
    {
        return lhs.get () == rhs.get ();
    }
    [[nodiscard]] friend constexpr bool
    operator!= (const OwnedIterator &lhs, const OwnedIterator &rhs) noexcept
    {
        return lhs.get () != rhs.get ();
    }

    constexpr OwnedIterator &
    operator++ () noexcept
    {
        ++this->_M_current;
        return *this;
    }
    constexpr OwnedIterator
    operator++ (int) noexcept
    {
        return OwnedIterator (this->_M_current++);
    }
    constexpr OwnedIterator &
    operator-- () noexcept
    {
        --this->_M_current;
        return *this;
    }
    constexpr OwnedIterator
    operator-- (int) noexcept
    {
        return OwnedIterator (_M_current--);
    }
    constexpr OwnedIterator &
    operator+= (difference_type n) noexcept
    {
        this->_M_current += n;
        return *this;
    }
    [[nodiscard]] constexpr OwnedIterator
    operator+ (difference_type n) const noexcept
    {
        return OwnedIterator (this->_M_current + n);
    }
    [[nodiscard]] constexpr difference_type
    operator+ (const OwnedIterator &other) const noexcept
    {
        return static_cast<difference_type> (this->get () + other.get ());
    }
    constexpr OwnedIterator &
    operator-= (difference_type n) noexcept
    {
        this->_M_current -= n;
        return *this;
    }
    [[nodiscard]] constexpr OwnedIterator
    operator- (difference_type n) const noexcept
    {
        return OwnedIterator (this->_M_current - n);
    }
    [[nodiscard]] constexpr difference_type
    operator- (const OwnedIterator &other) const noexcept
    {
        return static_cast<difference_type> (this->get () - other.get ());
    }
    template <typename Iterer>
    constexpr bool
    operator== (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (this->get () == other.get ()))
        requires requires {
            { this->get () == other.get () } -> std::convertible_to<bool>;
        }
    {
        return this->get () == other.get ();
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr SynthType<T, Iterer>
    operator<=> (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (_S_synth_three (this->get (), other.get ())))
    {
        return __synth (this->get (), other.get ());
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr SynthType<Iterer>
    operator<=> (const BasicIterator<Iterer> &other) const
        noexcept (noexcept (_S_synth_three (this->get (), other.get ())))
    {
        return _S_synth_three (this->get (), other.get ());
    }

    template <typename Iterer>
    [[nodiscard]]
    constexpr auto
    operator- (const BasicIterator<Iterer> &rhs) const noexcept
        -> decltype (this->get () - rhs.get ())
    {
        return this->get () - rhs.get ();
    }

    template <typename Iterer>
    [[nodiscard]] constexpr auto
    operator+ (difference_type n) const noexcept -> decltype (this->get () + n)
    {
        return BasicIterator<Iterer> (this->get () + n);
    }
    template <typename Iterer>
    [[nodiscard]] constexpr auto
    operator+ (const BasicIterator<Iterer> &other) const noexcept
        -> decltype (this->get () + other.get ())
    {
        return BasicIterator<Iterer> (this->get () + other.get ());
    }
    [[nodiscard]] constexpr OwnedIterator
    transfer () && noexcept
    {
        return std::move (*this);
    }
    [[nodiscard]] constexpr BorrowedIterator<T>
    borrow () & noexcept;
    [[nodiscard]] constexpr BorrowedIterator<const T>
    borrow () const & noexcept;

  private:
    const_pointer _M_current{};
};

template <typename T> class BorrowedIterator
{
  public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    constexpr BorrowedIterator () noexcept = default;
    explicit constexpr BorrowedIterator (const pointer value) noexcept
        : _M_current (value)
    {
    }
    explicit constexpr BorrowedIterator (const const_pointer value) noexcept
        : _M_current (value)
    {
    }
    explicit constexpr BorrowedIterator (reference value) noexcept
        : _M_current (value)
    {
    }
    explicit constexpr BorrowedIterator (const_reference value) noexcept
        : _M_current (value)
    {
    }
    BorrowedIterator (const BorrowedIterator &) noexcept = default;
    BorrowedIterator &
    operator= (const BorrowedIterator &) noexcept = default;

    [[nodiscard]] constexpr explicit
    operator bool () const noexcept
    {
        return this->get () != nullptr;
    }
    [[nodiscard]] constexpr pointer
    get () noexcept
    {
        return this->_M_current;
    }
    [[nodiscard]] constexpr const_pointer
    get () const noexcept
    {
        return this->_M_current;
    }
    [[nodiscard]] constexpr
    operator pointer () noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr
    operator const_pointer () const noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr reference
    operator* () noexcept
    {
        return *this->get ();
    }
    [[nodiscard]] constexpr const_reference
    operator* () const noexcept
    {
        return *this->get ();
    }
    [[nodiscard]] constexpr pointer
    operator->() noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr const_pointer
    operator->() const noexcept
    {
        return this->get ();
    }
    [[nodiscard]] constexpr reference
    operator[] (difference_type n) noexcept
    {
        return this->get ()[n];
    }
    [[nodiscard]] constexpr const_reference
    operator[] (difference_type n) const noexcept
    {
        return this->get ()[n];
    }
    constexpr pointer
    release () noexcept
    {
        return static_cast<pointer> (
            std::exchange (this->_M_current, pointer{}));
    }

    [[nodiscard]] constexpr BorrowedIterator
    operator+ (difference_type offset) const noexcept
    {
        return BorrowedIterator (this->_M_current + offset, this->__token);
    }
    [[nodiscard]] constexpr BorrowedIterator &
    operator+= (difference_type offset) noexcept
    {
        pointer value = this->get () + offset;
        this->_M_current = value;
        return *this;
    }
    [[nodiscard]] constexpr difference_type
    operator- (const BorrowedIterator &other) const noexcept
    {
        return this->get () - other.get ();
    }

    [[nodiscard]] constexpr bool
    operator== (std::nullptr_t) const noexcept
    {
        return this->get () == nullptr;
    }
    [[nodiscard]] constexpr bool
    operator!= (std::nullptr_t) const noexcept
    {
        return this->get () != nullptr;
    }
    [[nodiscard]] constexpr bool
    operator== (const BorrowedIterator &other) const noexcept
    {
        return this->get () == other.get ();
    }
    [[nodiscard]] constexpr bool
    operator!= (const BorrowedIterator &other) const noexcept
    {
        return !(*this == other);
    }

  private:
    struct __borrow_token
    {
    };
    constexpr BorrowedIterator (const pointer value,
                                __borrow_token token) noexcept
        : _M_current (value), __token (token)
    {
    }
    const_pointer _M_current{};
    [[no_unique_address]] __borrow_token __token{};

    template <typename, typename> friend class OwnedIterator;
};

template <typename T, typename Deleter>
constexpr BorrowedIterator<T>
OwnedIterator<T, Deleter>::borrow () & noexcept
{
    return BorrowedIterator<T> (this->get (),
                                typename BorrowedIterator<T>::__borrow_token{});
}

template <typename T, typename Deleter>
constexpr BorrowedIterator<const T>
OwnedIterator<T, Deleter>::borrow () const & noexcept
{
    return BorrowedIterator<const T> (
        this->get (), typename BorrowedIterator<const T>::__borrow_token{});
}

template <typename T> OwnedIterator (T *) -> OwnedIterator<T>;

template <typename T> inline constexpr bool IsOwnedPointer = false;

template <typename T, typename Deleter>
inline constexpr bool IsOwnedPointer<OwnedIterator<T, Deleter>> = true;

template <typename T> inline constexpr bool IsBorrowedPointer = false;

template <typename T>
inline constexpr bool IsBorrowedPointer<BorrowedIterator<T>> = true;

template <typename T>
concept OwnedPointerT = IsOwnedPointer<std::remove_cvref_t<T>>;

template <typename T>
concept BorrowedPointerT = IsBorrowedPointer<std::remove_cvref_t<T>>;
} // namespace lightzss::std