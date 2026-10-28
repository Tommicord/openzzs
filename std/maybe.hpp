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
#include <cstdint>
#include <functional>
#include <utility>

template <typename Tv, typename Ev> class MaybePayloadBase
{
    using value_type = std::remove_const_t<Tv>;
    using error_type = std::remove_const_t<Ev>;

  public:
    MaybePayloadBase () noexcept : _M_val_payload (), _M_engaged (false)
    {
    }
    ~MaybePayloadBase () noexcept = default;

    template <typename... Args>
    constexpr explicit MaybePayloadBase (std::in_place_t,
                                         Args &&...args) noexcept
        : _M_val_payload (std::in_place, std::forward<Args> (args)...),
          _M_engaged (true)
    {
    }

    template <typename ValUp, typename... Args>
    constexpr explicit MaybePayloadBase (std::initializer_list<ValUp> il,
                                         Args &&...args) noexcept
        : _M_val_payload (il, std::forward<Args> (args)...), _M_engaged (true)
    {
    }

    struct err_in_place_t
    {
    };
    template <typename... Args>
    constexpr explicit MaybePayloadBase (err_in_place_t,
                                         Args &&...args) noexcept
        : _M_err_payload (std::in_place, std::forward<Args> (args)...),
          _M_engaged (false), _M_err_engaged (true)
    {
    }
    constexpr MaybePayloadBase (const bool engaged_flag,
                                const MaybePayloadBase &other) noexcept
        : _M_val_payload (), _M_engaged (false)
    {
        if (engaged_flag)
            this->_M_construct (other._M_get ());
        else if (other._M_err_engaged)
            this->_M_construct_err (other._M_err_get ());
    }
    constexpr MaybePayloadBase (const bool engaged_flag,
                                MaybePayloadBase &&other) noexcept
        : _M_val_payload (), _M_engaged (false)
    {
        if (engaged_flag)
            this->_M_construct (std::move (other._M_get ()));
        else if (other._M_err_engaged)
            this->_M_construct_err (std::move (other._M_err_get ()));
    }

    MaybePayloadBase (const MaybePayloadBase &) = default;
    MaybePayloadBase (MaybePayloadBase &&) = default;
    MaybePayloadBase &
    operator= (const MaybePayloadBase &) = default;
    MaybePayloadBase &
    operator= (MaybePayloadBase &&) = default;

    constexpr void
    _S_copy_assign (const MaybePayloadBase &other) noexcept
    {
        if (this->_M_engaged && other._M_engaged)
            this->_M_get () = other._M_get ();
        else if (this->_M_err_engaged && other._M_err_engaged)
            this->_M_err_get () = other._M_err_get ();
        else if (other._M_engaged)
            {
                this->reset ();
                this->_M_construct (other._M_get ());
            }
        else if (other._M_err_engaged)
            {
                this->reset ();
                this->_M_construct_err (other._M_err_get ());
            }
        else
            this->reset ();
    }
    constexpr void
    _S_move_assign (MaybePayloadBase &&other) noexcept (
        std::is_nothrow_move_constructible_v<Tv>
        && std::is_nothrow_move_assignable_v<Tv>)
    {
        if (this->_M_engaged && other._M_engaged)
            this->_M_get () = std::move (other._M_get ());
        else if (this->_M_err_engaged && other._M_err_engaged)
            this->_M_err_get () = std::move (other._M_err_get ());
        else if (other._M_engaged)
            {
                this->reset ();
                this->_M_construct (std::move (other._M_get ()));
            }
        else if (other._M_err_engaged)
            {
                this->reset ();
                this->_M_construct_err (std::move (other._M_err_get ()));
            }
        else
            this->reset ();
    }
    struct _M_empty_byte
    {
    };

    template <typename ValUp, bool = std::is_trivially_destructible_v<ValUp>>
    union MaybeStorage
    {
        constexpr MaybeStorage () noexcept : _M_empty ()
        {
        }

        template <typename... Args>
        constexpr explicit MaybeStorage (std::in_place_t, Args &&...args)
            : _M_val (std::forward<Args> (args)...)
        {
        }

        template <typename V, typename... Args>
        constexpr explicit MaybeStorage (std::initializer_list<V> il,
                                         Args &&...args)
            : _M_val (il, std::forward<Args> (args)...)
        {
        }
        ~MaybeStorage () = default;

        ~MaybeStorage ()
            requires (!std::is_trivially_destructible_v<ValUp>)
        {
        }
        MaybeStorage (const MaybeStorage &) = default;
        MaybeStorage (MaybeStorage &&) = default;
        MaybeStorage &
        operator= (const MaybeStorage &) = default;
        MaybeStorage &
        operator= (MaybeStorage &&) = default;
        _M_empty_byte _M_empty;
        ValUp _M_val;
    };
    template <typename... Args>
    constexpr void
    _M_construct (Args &&...args) noexcept (
        std::is_nothrow_constructible_v<value_type, Args...>)
    {
        ::new (
            static_cast<void *> (std::addressof (this->_M_val_payload._M_val)))
            value_type (std::forward<Args> (args)...);
        this->_M_engaged = true;
    }
    template <typename... Args>
    constexpr void
    _M_construct_err (Args &&...args) noexcept (
        std::is_nothrow_constructible_v<error_type, Args...>)
    {
        ::new (
            static_cast<void *> (std::addressof (this->_M_err_payload._M_val)))
            error_type (std::forward<Args> (args)...);
        this->_M_err_engaged = true;
    }
    constexpr void
    _M_destruct () noexcept
    {
        this->_M_val_payload._M_val.~value_type ();
    }
    constexpr void
    _M_destruct_err () noexcept
    {
        this->_M_err_payload._M_val.~error_type ();
    }
    constexpr Tv &
    _M_get () noexcept
    {
        return this->_M_val_payload._M_val;
    }
    constexpr const Tv &
    _M_get () const noexcept
    {
        return this->_M_val_payload._M_val;
    }
    constexpr error_type &
    _M_err_get () noexcept
    {
        return this->_M_err_payload._M_val;
    }
    constexpr const error_type &
    _M_err_get () const noexcept
    {
        return this->_M_err_payload._M_val;
    }
    constexpr void
    reset () noexcept
    {
        if (this->_M_engaged)
            {
                this->_M_destruct ();
                this->_M_engaged = false;
            }
        if (this->_M_err_engaged)
            {
                this->_M_destruct_err ();
                this->_M_err_engaged = false;
            }
    }
    union
    {
        MaybeStorage<value_type> _M_val_payload;
        MaybeStorage<error_type> _M_err_payload;
    };
    bool _M_engaged;
    bool _M_err_engaged = false;
};
template <typename Tv, typename Ev,
          bool trivial_destruct = std::is_trivially_destructible_v<Tv>,
          bool trivial_copy = std::is_trivially_copy_assignable_v<Tv>
                              && std::is_trivially_copy_constructible_v<Tv>,
          bool trivial_move = std::is_trivially_move_assignable_v<Tv>
                              && std::is_trivially_move_constructible_v<Tv>>
struct MaybePayload;

template <typename Tv, typename Ev>
struct MaybePayload<Tv, Ev, true, true, true> : MaybePayloadBase<Tv, Ev>
{
    using MaybePayloadBase<Tv, Ev>::MaybePayloadBase;
    MaybePayload () = default;
};

template <typename Tv, typename Ev>
struct MaybePayload<Tv, Ev, true, false, true> : MaybePayloadBase<Tv, Ev>
{
    using MaybePayloadBase<Tv, Ev>::MaybePayloadBase;
    MaybePayload () = default;
    ~MaybePayload () = default;
    MaybePayload (const MaybePayload &) = default;
    MaybePayload (MaybePayload &&) = default;
    MaybePayload &
    operator= (MaybePayload &&) = default;

    constexpr MaybePayload &
    operator= (const MaybePayload &other)
    {
        this->_S_copy_assign (other);
        return *this;
    }
};

template <typename Tv, typename Ev>
struct MaybePayload<Tv, Ev, true, true, false> : MaybePayloadBase<Tv, Ev>
{
    using MaybePayloadBase<Tv, Ev>::MaybePayloadBase;
    MaybePayload () = default;
    ~MaybePayload () = default;
    MaybePayload (const MaybePayload &) = default;
    MaybePayload (MaybePayload &&) = default;
    MaybePayload &
    operator= (const MaybePayload &) = default;

    constexpr MaybePayload &
    operator= (MaybePayload &&other) noexcept (
        std::is_nothrow_move_constructible_v<Tv>
        && std::is_nothrow_move_assignable_v<Tv>)
    {
        this->_S_move_assign (std::move (other));
        return *this;
    }
};
template <typename Tv, typename Ev>
struct MaybePayload<Tv, Ev, true, false, false> : MaybePayloadBase<Tv, Ev>
{
    using MaybePayloadBase<Tv, Ev>::MaybePayloadBase;
    MaybePayload () = default;
    ~MaybePayload () = default;
    MaybePayload (const MaybePayload &) = default;
    MaybePayload (MaybePayload &&) = default;

    constexpr MaybePayload &
    operator= (const MaybePayload &other)
    {
        this->_S_copy_assign (other);
        return *this;
    }

    constexpr MaybePayload &
    operator= (MaybePayload &&other) noexcept (
        std::is_nothrow_move_constructible_v<Tv>
        && std::is_nothrow_move_assignable_v<Tv>)
    {
        this->_S_move_assign (std::move (other));
        return *this;
    }
};
template <typename Tv, typename Ev, bool copy, bool move>
struct MaybePayload<Tv, Ev, false, copy, move>
    : MaybePayload<Tv, Ev, true, false, false>
{
    using MaybePayload<Tv, Ev, true, false, false>::MaybePayload;
    MaybePayload () = default;
    MaybePayload (const MaybePayload &) = default;
    MaybePayload (MaybePayload &&) = default;
    MaybePayload &
    operator= (const MaybePayload &) = default;
    MaybePayload &
    operator= (MaybePayload &&) = default;
    constexpr ~MaybePayload ()
    {
        this->reset ();
    }
};
template <typename Tv, typename Ev> struct MaybeBase
{
    using value_type = Tv;
    using error_type = Ev;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    constexpr MaybeBase () = default;

    template <typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<Tv, Args...>>>
    constexpr explicit MaybeBase (std::in_place_t, Args &&...args)
        : _M_payload (std::in_place, std::forward<Args> (args)...)
    {
    }

    template <typename U, typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<
                  Tv, std::initializer_list<U> &, Args...>>>
    constexpr explicit MaybeBase (std::in_place_t, std::initializer_list<U> il,
                                  Args &&...args)
        : _M_payload (std::in_place, il, std::forward<Args> (args)...)
    {
    }

    template <typename U>
    constexpr explicit MaybeBase (std::in_place_t, U v)
        : _M_payload (std::in_place, std::forward<U> (v))
    {
    }

    using err_in_place_t = typename MaybePayloadBase<Tv, Ev>::err_in_place_t;
    template <typename U>
    constexpr explicit MaybeBase (err_in_place_t, U v)
        : _M_payload (err_in_place_t{}, std::forward<U> (v))
    {
    }

    constexpr MaybeBase (const MaybeBase &other) noexcept (
        std::is_nothrow_copy_constructible_v<Tv>)
        : _M_payload (other._M_payload._M_engaged, other._M_payload)
    {
    }

    constexpr MaybeBase (MaybeBase &&other) noexcept (
        std::is_nothrow_move_constructible_v<Tv>)
        : _M_payload (other._M_payload._M_engaged, std::move (other._M_payload))
    {
    }
    constexpr MaybeBase (const MaybeBase &)
        requires std::is_trivially_copy_constructible_v<Tv>
    = default;
    constexpr MaybeBase (MaybeBase &&)
        requires std::is_trivially_move_constructible_v<Tv>
    = default;
    MaybeBase &
    operator= (const MaybeBase &) = default;
    MaybeBase &
    operator= (MaybeBase &&) = default;
    MaybePayload<Tv, Ev> _M_payload;

    template <typename... Args>
    constexpr void
    construct (Args &&...args) noexcept (
        std::is_nothrow_constructible_v<std::remove_cvref_t<value_type>,
                                        Args...>)
    {
        _M_payload._M_construct (std::forward<Args> (args)...);
    }
    constexpr void
    destruct () noexcept
    {
        _M_payload._M_destruct ();
    }
    constexpr void
    reset () noexcept
    {
        _M_payload.reset ();
    }
    constexpr bool
    is_engaged () const noexcept
    {
        return _M_payload._M_engaged;
    }
    constexpr bool
    has_err () const noexcept
    {
        return _M_payload._M_err_engaged;
    }
    constexpr value_type &
    get () noexcept
    {
        return _M_payload._M_get ();
    }
    constexpr const value_type &
    get () const noexcept
    {
        return _M_payload._M_get ();
    }
    constexpr error_type &
    err () noexcept
    {
        return _M_payload._M_err_get ();
    }
    constexpr const error_type &
    err () const noexcept
    {
        return _M_payload._M_err_get ();
    }
    constexpr bool
    engaged () const noexcept
    {
        return is_engaged ();
    }
    constexpr value_type &
    stored () noexcept
    {
        return get ();
    }
    constexpr const value_type &
    stored () const noexcept
    {
        return get ();
    }
};

template <typename Signature> struct result_of;

struct invoke_memfun_ref
{
};
struct invoke_memfun_deref
{
};
struct invoke_memobj_ref
{
};
struct invoke_memobj_deref
{
};
struct invoke_other
{
};

template <typename Tv, typename ValUp>
constexpr ValUp &&
invfwd (std::remove_reference_t<Tv> &t) noexcept
{
    return static_cast<ValUp &&> (t);
}

template <typename Res, typename Fn, typename... Args>
constexpr Res
invoke_impl (invoke_other, Fn &&f, Args &&...args)
{
    return std::forward<Fn> (f) (std::forward<Args> (args)...);
}

template <typename Res, typename MemFun, typename Tv, typename... Args>
constexpr Res
invoke_impl (invoke_memfun_ref, MemFun &&f, Tv &&t, Args &&...args)
{
    return (invfwd<Tv> (t).*f) (std::forward<Args> (args)...);
}

template <typename Res, typename MemFun, typename Tv, typename... Args>
constexpr Res
invoke_impl (invoke_memfun_deref, MemFun &&f, Tv &&t, Args &&...args)
{
    return ((*std::forward<Tv> (t)).*f) (std::forward<Args> (args)...);
}

template <typename Res, typename MemPtr, typename Tv>
constexpr Res
invoke_impl (invoke_memobj_ref, MemPtr &&f, Tv &&t)
{
    return invfwd<Tv> (t).*f;
}

template <typename Res, typename MemPtr, typename Tv>
constexpr Res
invoke_impl (invoke_memobj_deref, MemPtr &&f, Tv &&t)
{
    return (*std::forward<Tv> (t)).*f;
}

template <typename Callable, typename... Args>
constexpr std::invoke_result_t<Callable, Args...>
invoke (Callable &&fn, Args &&...args) noexcept (
    std::is_nothrow_invocable_v<Callable, Args...>)
{
    using result_type = std::invoke_result<Callable, Args...>;
    using type = result_type::type;
    using tag = result_type::invoke_type;
    return invoke_impl<type> (tag{}, std::forward<Callable> (fn),
                              std::forward<Args> (args)...);
}

namespace zzs::base
{
template <typename CharTy, unsigned int N, typename Tv>
class MaybeHandleMessaged;

template <typename Tv> class MaybeHandle;

template <typename Tv, typename Ev> class Maybe : MaybeBase<Tv, Ev>
{
  public:
    using base_type = MaybeBase<Tv, Ev>;
    using value_type = Tv;
    using error_type = Ev;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static_assert (
        !std::is_same_v<std::remove_cv_t<value_type>, std::in_place_t>);
    static_assert (std::is_object_v<value_type>
                   && !std::is_array_v<value_type>);

    constexpr Maybe () noexcept = default;
    constexpr Maybe (nullptr_t) noexcept
    {
    }

    template <typename ValUp = std::remove_cv_t<value_type>>
        requires (!std::is_same_v<Maybe, std::remove_cvref_t<ValUp>>)
                 && (!std::is_same_v<std::in_place_t,
                                     std::remove_cvref_t<ValUp>>)
                 && std::is_constructible_v<value_type, ValUp>
    constexpr explicit (!std::is_convertible_v<ValUp, value_type>)
        Maybe (ValUp &&value) noexcept (
            std::is_nothrow_constructible_v<value_type, ValUp>)
        : base_type (std::in_place, std::forward<ValUp> (value))
    {
    }

    template <typename ValErr = std::remove_cv_t<error_type>>
        requires (!std::is_same_v<Maybe, std::remove_cvref_t<ValErr>>)
                 && (!std::is_same_v<std::in_place_t,
                                     std::remove_cvref_t<ValErr>>)
                 && std::is_constructible_v<error_type, ValErr>
                 && (!std::is_constructible_v<value_type, ValErr>)
    constexpr explicit (!std::is_convertible_v<ValErr, error_type>)
        Maybe (ValErr &&value) noexcept (
            std::is_nothrow_constructible_v<error_type, ValErr>)
        : base_type (typename base_type::err_in_place_t{},
                     std::forward<ValErr> (value))
    {
    }

    template <typename ValUp>
        requires (!std::is_same_v<value_type, ValUp>)
                 && std::is_constructible_v<value_type, const ValUp &>
    constexpr explicit (!std::is_convertible_v<const ValUp &, value_type>)
        Maybe (const Maybe<ValUp, Ev> &other) noexcept (
            std::is_nothrow_constructible_v<value_type, const ValUp &>)
        : base_type (other._M_payload._M_engaged, other._M_payload)
    {
    }

    template <typename ValUp>
        requires (!std::is_same_v<value_type, ValUp>)
                 && std::is_constructible_v<value_type, ValUp>
    constexpr explicit (!std::is_convertible_v<ValUp, value_type>)
        Maybe (Maybe<ValUp, Ev> &&other) noexcept (
            std::is_nothrow_constructible_v<value_type, ValUp>)
        : base_type (other._M_payload._M_engaged, std::move (other._M_payload))
    {
    }

    template <typename... Args>
        requires std::is_constructible_v<value_type, Args...>
    explicit constexpr Maybe (std::in_place_t, Args &&...args) noexcept (
        std::is_nothrow_constructible_v<value_type, Args...>)
        : base_type (std::in_place, std::forward<Args> (args)...)
    {
    }

    constexpr Maybe (const Maybe &other) noexcept (
        std::is_nothrow_copy_constructible_v<value_type>)
        requires std::is_copy_constructible_v<value_type>
    = default;
    constexpr Maybe (Maybe &&other) noexcept (
        std::is_nothrow_move_constructible_v<value_type>)
        requires std::is_move_constructible_v<value_type>
    = default;

    constexpr Maybe &
    operator= (const Maybe &other) noexcept (
        std::is_nothrow_copy_assignable_v<value_type>)
        requires std::is_copy_assignable_v<value_type>
    = default;
    constexpr Maybe &
    operator= (Maybe &&other) noexcept (
        std::is_nothrow_move_assignable_v<value_type>)
        requires std::is_move_assignable_v<value_type>
    = default;
    constexpr ~Maybe () = default;

    [[nodiscard]] constexpr bool
    has_value () const noexcept
    {
        return base_type::is_engaged ();
    }

    [[nodiscard]] constexpr explicit
    operator bool () const noexcept
    {
        return has_value ();
    }

    [[nodiscard]] constexpr bool
    has_error () const noexcept
    {
        return base_type::has_err ();
    }

    [[nodiscard]] constexpr error_type &
    error () & noexcept
    {
        return base_type::err ();
    }

    [[nodiscard]] constexpr const error_type &
    error () const & noexcept
    {
        return base_type::err ();
    }

    constexpr value_type &
    value () & noexcept
    {
        return base_type::get ();
    }

    constexpr const value_type &
    value () const & noexcept
    {
        return base_type::get ();
    }

    constexpr value_type &&
    value () && noexcept
    {
        return std::move (base_type::get ());
    }

    constexpr const value_type &&
    value () const && noexcept
    {
        return std::move (base_type::get ());
    }

    template <typename U>
        requires std::is_convertible_v<U, value_type>
    constexpr value_type
    value_or (U &&default_value) const &
    {
        return has_value ()
                   ? value ()
                   : static_cast<value_type> (std::forward<U> (default_value));
    }

    template <typename U>
        requires std::is_convertible_v<U, value_type>
    constexpr value_type
    value_or (U &&default_value) &&
    {
        return has_value ()
                   ? std::move (value ())
                   : static_cast<value_type> (std::forward<U> (default_value));
    }

    template <typename F>
        requires (std::invocable<F, value_type &>
                  && !std::is_void_v<std::invoke_result_t<F, value_type &>>)
    constexpr auto
    map (F &&f) & -> Maybe<
        std::remove_cvref_t<std::invoke_result_t<F, value_type &>>, Ev>
    {
        if (!has_value ())
            return {};
        else
            return std::invoke (std::forward<F> (f), value ());
    }

    template <typename F>
        requires (
            std::invocable<F, const value_type &>
            && !std::is_void_v<std::invoke_result_t<F, const value_type &>>)
    constexpr auto
    map (F &&f) const & -> Maybe<
        std::remove_cvref_t<std::invoke_result_t<F, const value_type &>>, Ev>
    {
        if (!has_value ())
            return {};
        else
            return std::invoke (std::forward<F> (f), value ());
    }

    template <typename F>
        requires (std::invocable<F, value_type &&>
                  && !std::is_void_v<std::invoke_result_t<F, value_type &&>>)
    constexpr auto
    map (F &&f) && -> Maybe<
        std::remove_cvref_t<std::invoke_result_t<F, value_type &&>>, Ev>
    {
        if (!has_value ())
            return {};
        else
            return std::invoke (std::forward<F> (f), std::move (value ()));
    }

    template <typename F>
    constexpr auto
    transform (F &&f) & -> decltype (this->map (std::forward<F> (f)))
    {
        return this->map (std::forward<F> (f));
    }

    template <typename F>
    constexpr auto
    transform (F &&f) const & -> decltype (this->map (std::forward<F> (f)))
    {
        return this->map (std::forward<F> (f));
    }

    template <typename F>
    constexpr auto
    transform (
        F &&f) && -> decltype (std::move (*this).map (std::forward<F> (f)))
    {
        return std::move (*this).map (std::forward<F> (f));
    }

    template <typename F>
        requires std::invocable<F, value_type &>
                 && requires (std::invoke_result_t<F, value_type &> result) {
                        { result.has_value () } -> std::convertible_to<bool>;
                    }
    constexpr auto
    and_then (
        F &&f) & -> std::remove_cvref_t<std::invoke_result_t<F, value_type &>>
    {
        using result_type =
            std::remove_cvref_t<std::invoke_result_t<F, value_type &>>;
        if (!this->has_value ())
            return result_type{};
        else
            return std::invoke (std::forward<F> (f), this->value ());
    }

    template <typename F>
        requires std::invocable<F, const value_type &>
                 && requires (
                     std::invoke_result_t<F, const value_type &> result) {
                        { result.has_value () } -> std::convertible_to<bool>;
                    }
    constexpr auto
    and_then (F &&f) const
        & -> std::remove_cvref_t<std::invoke_result_t<F, const value_type &>>
    {
        using result_type =
            std::remove_cvref_t<std::invoke_result_t<F, const value_type &>>;
        if (!this->has_value ())
            return result_type{};
        else
            return std::invoke (std::forward<F> (f), this->value ());
    }

    template <typename F>
        requires std::invocable<F, value_type &&>
                 && requires (std::invoke_result_t<F, value_type &&> result) {
                        { result.has_value () } -> std::convertible_to<bool>;
                    }
    constexpr auto
    and_then (
        F &&f) && -> std::remove_cvref_t<std::invoke_result_t<F, value_type &&>>
    {
        using result_type =
            std::remove_cvref_t<std::invoke_result_t<F, value_type &&>>;
        if (!this->has_value ())
            return result_type{};
        else
            return std::invoke (std::forward<F> (f),
                                std::move (this->value ()));
    }

    template <typename U>
        requires std::constructible_from<value_type, U>
    constexpr Maybe
    or_value (U &&value) const &
    {
        return this->has_value () ? Maybe (value ())
                                  : Maybe (std::forward<U> (value));
    }

    template <typename U>
        requires std::constructible_from<value_type, U>
    constexpr auto
    or_value (U &&value) &&
    {
        return this->has_value () ? Maybe (std::move (value ()))
                                  : Maybe (std::forward<U> (value));
    }

    constexpr auto
    or_value (Maybe other) const &
    {
        return this->has_value () ? Maybe (this->value ()) : std::move (other);
    }

    constexpr auto
    or_value (Maybe other) &&
    {
        return this->has_value () ? Maybe (std::move (this->value ()))
                                  : std::move (other);
    }

    template <typename F>
        requires std::invocable<F>
                 && std::constructible_from<value_type, std::invoke_result_t<F>>
    constexpr auto
    or_else (F &&f) const &
    {
        if (this->has_value ())
            return Maybe<value_type, error_type> (this->value ());
        else
            return Maybe<value_type, error_type> (
                std::invoke (std::forward<F> (f)));
    }

    template <typename U>
    constexpr auto
    and_other (Maybe<U, Ev> other) const &
    {
        return this->has_value () ? std::move (other) : Maybe<U, Ev>{};
    }

    template <typename U>
    constexpr auto
    and_other (Maybe<U, Ev> other) &&
    {
        return this->has_value () ? std::move (other) : Maybe<U, Ev>{};
    }

    [[nodiscard]] constexpr bool
    ok () const noexcept
    {
        return this->has_value ();
    }

    template <typename E>
        requires (std::is_enum_v<E>)
    constexpr auto
    ok_or (E &&err) const &
    {
        using maybe_type = MaybeHandle<value_type>;
        if (this->has_value ())
            return maybe_type (this->value ());
        else
            return maybe_type (err);
    }

    template <typename E>
    constexpr auto
    ok_or (E &&err) &&
    {
        using maybe_type = MaybeHandle<value_type>;
        if (this->has_value ())
            return maybe_type (std::move (this->value ()));
        else
            return maybe_type (std::forward<E> (err));
    }

    template <typename F>
        requires (std::invocable<F, Tv &>
                  && !std::is_void_v<std::invoke_result_t<F, Tv &>>)
    constexpr auto
    map (F &&f)
        & -> MaybeHandle<std::remove_cvref_t<std::invoke_result_t<F, Tv &>>>
    {
        if (!this->has_value ())
            return MaybeHandle (this->what ());
        else
            return MaybeHandle (invoke (std::forward<F> (f), this->value ()));
    }

    template <typename F>
        requires (std::invocable<F, const Tv &>
                  && !std::is_void_v<std::invoke_result_t<F, const Tv &>>)
    constexpr auto
    map (F &&f) const & -> MaybeHandle<
        std::remove_cvref_t<std::invoke_result_t<F, const Tv &>>>
    {
        if (!this->has_value ())
            return MaybeHandle (this->what ());
        else
            return MaybeHandle (invoke (std::forward<F> (f), this->value ()));
    }

    template <typename F>
        requires (std::invocable<F, Tv &&>
                  && !std::is_void_v<std::invoke_result_t<F, Tv &&>>)
    constexpr auto
    map (F &&f)
        && -> MaybeHandle<std::remove_cvref_t<std::invoke_result_t<F, Tv &&>>>
    {
        if (!this->has_value ())
            return MaybeHandle (this->what ());
        else
            return MaybeHandle (
                invoke (std::forward<F> (f), std::move (this->value ())));
    }

    template <typename U>
        requires std::constructible_from<Tv, U>
    constexpr auto
    or_value (U &&value) const &
    {
        return has_value () ? MaybeHandle (value ())
                            : MaybeHandle (std::forward<U> (value));
    }

    template <typename U>
    constexpr auto
    and_other (MaybeHandle<U> other) const &
    {
        return this->has_value () ? std::move (other) : Maybe{};
    }

    [[nodiscard]] constexpr const char *
    what () const noexcept
    {
        return this->_M_payload.get ();
    }

    template <typename E>
    constexpr auto
    ok_or (E &&err) const &
    {
        if (this->has_value ())
            return Maybe (this->value ());
        else
            return err;
    }

    template <typename U>
        requires std::is_convertible_v<U, value_type>
    constexpr value_type
    unroll_or (U &&default_value) const &
    {
        static_assert (std::is_copy_constructible_v<value_type>
                           || std::is_move_constructible_v<Tv>,
                       "Tv must be copy or move constructible for unroll_or");
        return this->has_value ()
                   ? this->value ()
                   : static_cast<Tv> (std::forward<U> (default_value));
    }

    template <typename ValUp>
        requires std::is_convertible_v<ValUp, Tv>
    constexpr value_type
    unroll_or (ValUp &&default_value) &&
    {
        static_assert (std::is_copy_constructible_v<value_type>
                           || std::is_move_constructible_v<Tv>,
                       "Tv must be copy or move constructible for unroll_or");
        return this->has_value () ? std::move (this->value ())
                                  : static_cast<value_type> (
                                        std::forward<ValUp> (default_value));
    }

    template <typename F>
        requires std::invocable<F>
    constexpr void
    unroll_or_else (F &&f) const &
    {
        if (!this->has_value ())
            invoke (std::forward<F> (f));
    }

    template <typename F>
        requires std::invocable<F>
    constexpr void
    unroll_or_else (F &&f) &&
    {
        if (!this->has_value ())
            invoke (std::forward<F> (f));
    }

    template <typename ValFn>
        requires std::invocable<ValFn>
                 && std::is_convertible_v<decltype (std::declval<ValFn> () ()),
                                          Tv>
    constexpr value_type
    unroll_or_else (ValFn &&fn) const &
    {
        if (this->has_value ())
            return this->value ();
        else
            return std::forward<ValFn> (fn) ();
    }

    template <typename Fn>
        requires std::invocable<Fn>
                 && std::is_convertible_v<decltype (std::declval<Fn> () ()), Tv>
    constexpr value_type
    unroll_or_else (Fn &&fn) &&
    {
        if (this->has_value ())
            return std::move (this->value ());
        else
            return std::forward<Fn> (fn) ();
    }

    template <typename ValEn>
        requires (std::is_enum_v<ValEn>
                  || std::is_base_of_v<ValEn, MaybeHandle<Tv>>)
    constexpr auto
    unroll_map () const noexcept -> MaybeHandle<ValEn>
    {
        static_assert (std::is_enum_v<ValEn> || std::is_class_v<ValEn>,
                       "zzs::base::MaybeHandle target type must be an "
                       "enum or class type");
        if (has_value ())
            return MaybeHandle (static_cast<ValEn> (this->value ()));
        else
            return MaybeHandle (this->what ());
    }
    constexpr void
    reset () noexcept
    {
        base_type::reset ();
    }
};

template <typename Tv> class MaybeErrorStorage
{
    using value_type = Tv;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

  public:
    bool _M_engaged = false;
    // Self-contained byte buffer (one spare byte for the trailing '\0' written
    // below). Deliberately not StoreBaseSingle<Tv>: memoryAllocator
    // imports this module (Maybe appears in the allocator APIs), so importing
    // it back would be an import cycle that CMake rejects.
    Tv _M_storage[sizeof (Tv) + 1]{};
    constexpr MaybeErrorStorage () noexcept = default;
    explicit constexpr MaybeErrorStorage (const Tv v) noexcept
        : _M_engaged (true)
    {
        struct Uint8TBytes
        {
            uint8_t b[sizeof (Tv)];
        };
        auto valty_bytes = std::bit_cast<Uint8TBytes> (v);
        for (size_type i = 0; i < sizeof (Tv); ++i)
            this->_M_storage[i] = valty_bytes.b[i];
        this->_M_storage[sizeof (Tv)] = '\0';
    }

    constexpr MaybeErrorStorage (const MaybeErrorStorage &other) noexcept
        : _M_engaged (other._M_engaged)
    {
        this->_M_copy_from (other);
    }

    constexpr MaybeErrorStorage (MaybeErrorStorage &&other) noexcept
        : _M_engaged (other._M_engaged)
    {
        this->_M_copy_bytes (other);
        other._M_engaged = false;
    }
    constexpr MaybeErrorStorage &
    operator= (const MaybeErrorStorage &other) noexcept
    {
        if (std::addressof (other) != this)
            this->_M_copy_from (other);
        return *this;
    }
    constexpr MaybeErrorStorage &
    operator= (MaybeErrorStorage &&other) noexcept
    {
        if (std::addressof (other) != this)
            {
                this->_M_engaged = other._M_engaged;
                this->_M_copy_bytes (other);
                other._M_engaged = false;
            }
        return *this;
    }
    constexpr ~MaybeErrorStorage () noexcept = default;
    constexpr char *
    get () noexcept
    {
        return this->_M_engaged ? this->_M_storage : nullptr;
    }
    constexpr const char *
    get () const noexcept
    {
        return this->_M_engaged ? this->_M_storage : nullptr;
    }
    constexpr bool
    error_state () const noexcept
    {
        return this->_M_engaged;
    }

  private:
    constexpr void
    _M_copy_bytes (const MaybeErrorStorage &other) noexcept
    {
        for (size_type i = 0; i < sizeof (this->_M_storage); ++i)
            this->_M_storage[i] = other._M_storage[i];
    }
    constexpr void
    _M_copy_from (const MaybeErrorStorage &other) noexcept
    {
        this->_M_engaged = other._M_engaged;
        this->_M_copy_bytes (other);
    }
};
} // namespace zzs::base