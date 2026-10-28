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

#include <cinttypes>
#include <concepts>
#include <memory>
#include <new>
#include <type_traits>

namespace zzs::base
{
template <typename T, int Sz> struct StoreBase
{
    static_assert (Sz > 0, "zzs::base::BackingStorage invalid capacity");
    static_assert (
        !std::is_reference_v<T>,
        "zzs::base::BackingStorage required a non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::BackingStorage cannot allocate void");
    static_assert (
        std::is_destructible_v<T>,
        "zzs::base::BackingStorage required a destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::BackingStorage required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::BackingStorage required a trivially "
                   "copyable value type");
    static_assert (Sz > 0,
                   "zzs::base::BackingStorage capacity must be valid");
    alignas (alignof (T)) T _M_storage[Sz]{};

    StoreBase () noexcept;
    StoreBase (const StoreBase &other) noexcept = default;
    StoreBase (StoreBase &&) noexcept = delete;
    ~StoreBase () noexcept;

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static constexpr size_type capacity_value = Sz;
    pointer
    data () noexcept
    {
        return this->_M_storage;
    }
    const_pointer
    data () const noexcept
    {
        return this->_M_storage;
    }
    reference
    operator[] (const size_type index) noexcept
    {
        return this->_M_storage[index];
    }
    const_reference
    operator[] (const size_type index) const noexcept
    {
        return this->_M_storage[index];
    }
};
template <typename T> struct StoreBaseSingle
{
    static_assert (!std::is_reference_v<T>,
                   "zzs::base::StoreBaseSingle required a "
                   "non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::StoreBaseSingle cannot allocate void");
    static_assert (std::is_destructible_v<T>,
                   "zzs::base::StoreBaseSingle required a "
                   "destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::StoreBaseSingle required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::StoreBaseSingle required a trivially "
                   "copyable value type");
    alignas (alignof (T)) T _M_storage{};

    StoreBaseSingle () noexcept;
    StoreBaseSingle (const StoreBaseSingle &other) noexcept = default;
    StoreBaseSingle (StoreBaseSingle &&) noexcept = delete;
    ~StoreBaseSingle () noexcept;

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    pointer
    data () noexcept
    {
        return this->_M_storage;
    }
    const_pointer
    data () const noexcept
    {
        return this->_M_storage;
    }
    reference
    operator[] (const size_type index) noexcept
    {
        return this->_M_storage[index];
    }
    const_reference
    operator[] (const size_type index) const noexcept
    {
        return this->_M_storage[index];
    }
};

class AllocError
{
  public:
    enum class Kind
    {
        ZERO_SIZE, // allocate() was asked for zero elements
        TOO_LARGE, // the request cannot fit in the pool even when it is empty
        OUT_OF_MEMORY, // no free block large enough (pool exhausted or
                       // fragmented)
        INVALID_BLOCK, // Invalid block address
    };

    AllocError () = delete;
    explicit constexpr AllocError (const Kind kind) noexcept : _M_kind (kind)
    {
    }
    [[nodiscard]] constexpr Kind
    kind () const noexcept
    {
        return this->_M_kind;
    }
    [[nodiscard]] constexpr bool
    operator== (const AllocError &other) const noexcept
    {
        return this->_M_kind == other._M_kind;
    }

  private:
    Kind _M_kind;
};

template <class T, class Alloc, unsigned int Sz>
concept ContainerAllocator = requires {
    Sz > 0 &&
            requires (Alloc & alloc, typename Alloc::pointer p)
    {
        typename Alloc::value_type;
        typename Alloc::pointer;
        typename Alloc::size_type;
        requires std::same_as<typename Alloc::value_type, T>;
        {alloc.template allocate<Sz> ()}->std::same_as<typename Alloc::pointer>;
        {
            alloc.deallocate (p)
        } noexcept;
    };
};
template <class Alloc>
    requires ContainerAllocator<typename Alloc::value_type, Alloc,
                                Alloc::capacity_value>
struct AllocatorBase
{
    using allocator_type = Alloc;
    using value_type = typename Alloc::value_type;
    using pointer = typename Alloc::pointer;
    using const_pointer = typename Alloc::const_pointer;
    using reference = typename Alloc::reference;
    using const_reference = typename Alloc::const_reference;
    using size_type = typename Alloc::size_type;
    using difference_type = typename Alloc::difference_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;

    struct rebind
    {
        using other = AllocatorBase;
    };
    static void
    on_swap (allocator_type &a, allocator_type &b)
    {
        if (propagate_on_swap ())
            {
                auto temp = a;
                a = b;
                b = temp;
            }
    }

    template <typename _T, typename... Args>
        requires (!std::is_unbounded_array_v<_T>) && requires {
            ::new (static_cast<void *> (nullptr)) _T (std::declval<Args> ()...);
        }
    static _T *
    _S_construct_at (_T *location, Args &&...args) noexcept (noexcept (
        ::new (static_cast<void *> (nullptr)) _T (std::declval<Args> ()...)))
    {
        void *loc = location;
        if (std::is_array_v<_T>)
            {
                static_assert (
                    sizeof...(Args) == 0,
                    "zzs::base::container::alloc_traits::_S_construct_at "
                    "for array "
                    "types must not use any arguments to initialize the "
                    "array");
                return ::new (loc) _T[1]();
            }
        else
            return ::new (loc) _T (std::forward<Args> (args)...);
    }
    template <typename _T>
    static void
    _S_destroy_at (_T *location)
    {
        if (std::is_array_v<_T>)
            {
                for (auto &x : *location)
                    _S_destroy_at (std::addressof (x));
            }
        else
            location->~_T ();
    }
    template <typename ValUp, typename... Args>
    static void
    construct (ValUp *p, Args &&...args) noexcept (
        std::is_nothrow_constructible_v<ValUp, Args...>)
    {
        _S_construct_at (p, std::forward<Args> (args)...);
    }

    static allocator_type
    select_on_container_copy_construction (const allocator_type &alloc) noexcept
    {
        return alloc;
    }

    template <typename ValUp>
    static void
    destroy (ValUp *p) noexcept (std::is_nothrow_destructible_v<ValUp>)
    {
        _S_destroy_at (p);
    }
    static size_type
    max_size () noexcept
    {
        return static_cast<size_type> (-1) / sizeof (value_type);
    }

    static bool
    propagate_on_copy_assign ()
    {
        return propagate_on_container_copy_assignment::value;
    }

    static bool
    propagate_on_move_assign ()
    {
        return propagate_on_container_move_assignment::value;
    }

    static bool
    propagate_on_swap ()
    {
        return propagate_on_container_swap::value;
    }

    static bool
    always_equal ()
    {
        return is_always_equal::value;
    }

    static pointer
    allocate (Alloc &alloc, size_type count) noexcept
    {
        return alloc.allocate (count);
    }
    template <size_type N>
    static pointer
    allocate (Alloc &alloc) noexcept
    {
        if (requires { alloc.template allocate<N> (); })
            return alloc.template allocate<N> ();
        else
            return alloc.allocate (N);
    }
    static void
    deallocate (Alloc &alloc, pointer p) noexcept
    {
        alloc.deallocate (p);
    }
    template <class T, class... Args>
        requires std::constructible_from<T, Args...>
    static void
    construct (Alloc &, T *p, Args &&...args) noexcept (
        std::is_nothrow_constructible_v<T, Args...>)
    {
        _S_construct_at (p, std::forward<Args> (args)...);
    }
    template <class T>
    static void
    destroy (Alloc &, T *p) noexcept (std::is_nothrow_destructible_v<T>)
    {
        _S_destroy_at (p);
    }
};

template <class T, unsigned int Sz> class BumpAllocator
{
  public:
    static_assert (
        !std::is_reference_v<T>,
        "zzs::base::BumpAllocator required a non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::BumpAllocator cannot allocate void");
    static_assert (
        std::is_destructible_v<T>,
        "zzs::base::BumpAllocator required a destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::BumpAllocator required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::BumpAllocator required a trivially "
                   "copyable value type");
    static_assert (Sz > 0,
                   "zzs::base::BumpAllocator capacity must be valid");

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using iterator = BasicIterator<value_type>;
    using const_iterator = const BasicIterator<value_type>;
    using storage_type = StoreBase<T, Sz>;
    static constexpr size_type capacity_value = Sz;

    BumpAllocator &
    operator= (const BumpAllocator &) noexcept = delete;
    BumpAllocator (const BumpAllocator &other) noexcept = default;
    BumpAllocator (BumpAllocator &&other) noexcept = delete;
    BumpAllocator &
    operator= (BumpAllocator &&other) noexcept = delete;
    BumpAllocator () noexcept : _M_offset (0)
    {
    }

    template <unsigned int N>
        requires (N > 0 && N <= Sz)
    Maybe<pointer, AllocError>
    allocate () noexcept;
    Maybe<pointer, AllocError>
    allocate (const size_type count) noexcept;
    void deallocate (pointer) noexcept;
    ~BumpAllocator () noexcept;
    iterator
    begin () const noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    const_iterator
    begin () noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    iterator
    end () noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    const_iterator
    end () const noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    storage_type
    raw () const noexcept
    {
        return this->_M_storage;
    }
    const_iterator
    cbegin () const noexcept
    {
        return static_cast<const_iterator> (this->begin ());
    }
    const_iterator
    cend () const noexcept
    {
        return static_cast<const_iterator> (this->end ());
    }
    size_type
    max_size () const noexcept
    {
        return static_cast<size_type> (-1) / sizeof (value_type);
    }
    size_type
    used () const noexcept
    {
        return this->_M_offset * sizeof (value_type);
    }
    size_type
    capacity () const noexcept
    {
        return static_cast<size_type> (Sz) * sizeof (value_type);
    }
    bool
    is_empty () const noexcept
    {
        return this->_M_offset == 0;
    }
    void
    reset () const;

  private:
    storage_type _M_storage{};
    mutable size_type _M_offset = 0;
};

template <class T, unsigned int Sz> class BitmapAllocator
{
  public:
    static_assert (
        !std::is_reference_v<T>,
        "zzs::base::BitmapAllocator required a non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::BitmapAllocator cannot allocate void");
    static_assert (
        std::is_destructible_v<T>,
        "zzs::base::BitmapAllocator required a destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::BitmapAllocator required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::BitmapAllocator required a trivially "
                   "copyable value type");
    static_assert (Sz > 0,
                   "zzs::base::BitmapAllocator capacity must be valid");

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using iterator = BasicIterator<value_type>;
    using const_iterator = const BasicIterator<value_type>;
    using storage_type = StoreBase<T, Sz>;
    static constexpr size_type capacity_value = Sz;
    static constexpr size_type bitmap_words = ((Sz + 63) >> 6);
    static constexpr size_type max_bitmaps = Sz;

    BitmapAllocator &
    operator= (const BitmapAllocator &) noexcept = delete;
    BitmapAllocator (const BitmapAllocator &other) noexcept = default;
    BitmapAllocator (BitmapAllocator &&other) noexcept = delete;
    BitmapAllocator &
    operator= (BitmapAllocator &&other) noexcept = delete;
    BitmapAllocator () noexcept {};
    ~BitmapAllocator () noexcept {};

    template <unsigned int N>
    Maybe<pointer, AllocError>
    allocate () noexcept;
    void
    deallocate (pointer ptr) noexcept;
    iterator
    begin () const noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    const_iterator
    begin () noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    iterator
    end () noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    const_iterator
    end () const noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    storage_type
    raw () const noexcept
    {
        return this->_M_storage;
    }
    const_iterator
    cbegin () const noexcept
    {
        return static_cast<const_iterator> (this->begin ());
    }
    const_iterator
    cend () const noexcept
    {
        return static_cast<const_iterator> (this->end ());
    }
    size_type
    max_size () const noexcept
    {
        return static_cast<size_type> (-1) / sizeof (value_type);
    }
    size_type
    used () const noexcept
    {
        return this->_M_bitmap_count * sizeof (value_type);
    }
    size_type
    capacity () const noexcept
    {
        return static_cast<size_type> (Sz) * sizeof (value_type);
    }
    bool
    is_empty () const noexcept
    {
        return this->_M_bitmap_count == 0;
    }
    void
    reset () const noexcept;

  private:
    template <unsigned int N>
    pointer
    _S_handle_needed (size_type word_idx, size_type next_word,
                      size_type start_bit, size_type bits_needed,
                      size_type available) noexcept;
    storage_type _M_storage{};
    mutable uint64_t _M_allocation_bitmap[bitmap_words]{};
    mutable size_type _M_bitmap_count = 0;
};

template <class T, unsigned int Sz> class BuddyAllocator
{
  public:
    static_assert (
        !std::is_reference_v<T>,
        "zzs::base::BuddyAllocator required a non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::BuddyAllocator cannot allocate void");
    static_assert (
        std::is_destructible_v<T>,
        "zzs::base::BuddyAllocator required a destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::BuddyAllocator required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::BuddyAllocator required a trivially "
                   "copyable value type");
    static_assert (Sz > 0,
                   "zzs::base::BuddyAllocator capacity must be valid");

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using iterator = BasicIterator<value_type>;
    using const_iterator = const BasicIterator<value_type>;
    using storage_type = StoreBase<T, Sz>;
    using index_type = uint16_t;
    static constexpr size_type capacity_value = Sz;
    static constexpr index_type uninit = static_cast<index_type> (-1);

    BuddyAllocator &
    operator= (const BuddyAllocator &) noexcept = delete;
    BuddyAllocator (const BuddyAllocator &other) noexcept = default;
    BuddyAllocator (BuddyAllocator &&other) noexcept = delete;
    BuddyAllocator &
    operator= (BuddyAllocator &&other) noexcept = delete;
    BuddyAllocator () noexcept;

    template <unsigned int N>
    Maybe<pointer, AllocError>
    allocate () noexcept;
    Maybe<pointer, AllocError>
    allocate (size_type count) noexcept;
    void
    deallocate (pointer p) noexcept;
    ~BuddyAllocator () noexcept;
    iterator
    begin () const noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    const_iterator
    begin () noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    iterator
    end () noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    const_iterator
    end () const noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    storage_type
    raw () const noexcept
    {
        return this->_M_storage;
    }
    const_iterator
    cbegin () const noexcept
    {
        return static_cast<const_iterator> (this->begin ());
    }
    const_iterator
    cend () const noexcept
    {
        return static_cast<const_iterator> (this->end ());
    }
    size_type
    max_size () const noexcept
    {
        return static_cast<size_type> (-1) / sizeof (value_type);
    }
    size_type
    used () const noexcept
    {
        return this->_M_used_bytes;
    }
    size_type
    capacity () const noexcept
    {
        return static_cast<size_type> (Sz) * sizeof (value_type);
    }
    bool
    is_empty () const noexcept
    {
        return this->_M_used_bytes == 0;
    }
    void
    reset () const;

  private:
    static constexpr size_type
    _S_min_block_size () noexcept
    {
        size_type value_size = sizeof (value_type);
        size_type align = (63 << 1) + 32;
        if (value_size < align)
            return align;
        else
            return value_size;
    }
    static constexpr size_type
    _S_max_order () noexcept
    {
        size_type min_block_size = _S_min_block_size ();
        size_type size = Sz * sizeof (value_type);
        size_type max_order = 0;
        while ((static_cast<size_type> (1) << max_order) * min_block_size
               < size)
            ++max_order;
        return max_order;
    }
    static constexpr size_type
    _S_max_block_count () noexcept
    {
        const size_type order = _S_max_order ();
        if (constexpr size_type size_bits = sizeof (size_type) << 3;
            order >= size_bits - 1)
            return static_cast<size_type> (-1);
        else
            return static_cast<size_type> (1) << order;
    }
    static constexpr size_type _S_max_blocks = _S_max_block_count ();
    static constexpr size_type _S_free_lists = _S_max_order () + 1;

    storage_type _M_storage{};
    struct _S_BlockOwner
    {
        index_type _M_begin;
        index_type _M_end;
        index_type _M_next_block;

        _S_BlockOwner () = default;
        _S_BlockOwner (_S_BlockOwner &&) = delete;
        _S_BlockOwner (const _S_BlockOwner &other)
            : _M_begin (other._M_begin), _M_end (other._M_end),
              _M_next_block (other._M_next_block) {};
        _S_BlockOwner &
        operator= (_S_BlockOwner &&) = delete;
        _S_BlockOwner &
        operator= (const _S_BlockOwner &) = default;
        explicit _S_BlockOwner (const size_type begin,
                                const size_type end) noexcept
            : _M_begin (static_cast<index_type> (begin)),
              _M_end (static_cast<index_type> (end)), _M_next_block (uninit)
        {
        }
        size_type
        _S_size () const noexcept
        {
            return static_cast<size_type> (this->_M_end - this->_M_begin)
                   * sizeof (value_type);
        }
        bool
        _S_is_valid () const noexcept
        {
            return this->_M_begin != uninit && this->_M_end != uninit;
        }
        bool
        _S_is_allocated () const noexcept
        {
            return this->_S_is_valid ();
        }
        void
        _S_mark_deallocated () noexcept
        {
            this->_M_begin = this->_M_end = uninit;
        }
        void
        _S_initialize (const size_type begin, const size_type end) noexcept
        {
            this->_M_begin = static_cast<index_type> (begin);
            this->_M_end = static_cast<index_type> (end);
            this->_M_next_block = uninit;
        }
    };
    struct _S_BuddyState
    {
        static constexpr index_type uninit = static_cast<index_type> (-1);
        _S_BlockOwner _M_block_storage[_S_max_blocks]{};
        index_type _M_free_heads[_S_free_lists]{};
        index_type _M_block_count;
        index_type _M_free_head;

        _S_BuddyState () noexcept : _M_free_head (0)
        {
            this->_S_init_free_lists<0, _S_free_lists> ();
        }
        template <size_type Start, size_type End>
        void
        _S_init_free_lists () noexcept
        {
            if constexpr (Start < End)
                {
                    this->_M_free_heads[Start] = uninit;
                    this->_S_init_free_lists<Start + 1, End> ();
                }
        }
        _S_BuddyState (_S_BuddyState &&other) noexcept = default;
        _S_BuddyState (const _S_BuddyState &other) noexcept
        {
            for (size_type i = 0; i < _S_max_blocks; ++i)
                this->_M_block_storage[i] = other._M_block_storage[i];
            for (size_type i = 0; i < _S_free_lists; ++i)
                this->_M_free_heads[i] = other._M_free_heads[i];
            this->_M_block_count = other._M_block_count;
            this->_M_free_head = other._M_free_head;
        }
        _S_BuddyState
        operator= (const _S_BuddyState &) noexcept = delete;
        _S_BuddyState
        operator= (const _S_BuddyState &&) noexcept = delete;
        ~_S_BuddyState () noexcept = default;
    };
    void
    _S_construct_allocator () noexcept;
    mutable size_type _M_used_bytes = 0;
    mutable _S_BuddyState _M_buddy{};

    static bool
    _S_is_valid_block (const _S_BlockOwner *block) noexcept;
    static bool
    _S_is_block_allocated (const _S_BlockOwner *block) noexcept;
    template <size_type Size>
    static size_type
    _S_order_of_size () noexcept
    {
        static_assert (Size > 0,
                       "zzs::base::BuddyAllocator invalid size");
        size_type order = 0;
        size_type min_block_size = _S_min_block_size ();
        const size_type max_order = _S_max_order ();
        size_type block_size = min_block_size;
        while (block_size < Size && order < max_order)
            {
                block_size <<= 1;
                ++order;
            }
        return order;
    }
    template <size_type Start, size_type End, size_type Order,
              size_type BlockIndex = 0>
    size_type
    _S_do_find_free_block () noexcept
    {
        if (Start < End)
            {
                size_type curr_order = End - 1 - Start;
                if (const size_type free_head =
                        this->_M_buddy._M_free_heads[Start];
                    free_head == BlockIndex)
                    {
                        this->_S_remove_from_free_list<BlockIndex,
                                                       curr_order> ();
                        if (curr_order > Order)
                            this->_S_split_block<BlockIndex, curr_order,
                                                 Order> ();
                        return BlockIndex;
                    }
                if (BlockIndex + 1 < _S_max_blocks)
                    return this->_S_do_find_free_block<Start, End, Order,
                                                       BlockIndex + 1> ();
                else
                    return this
                        ->_S_do_find_free_block<Start + 1, End, Order, 0> ();
            }
        return _S_BuddyState::uninit;
    }
    template <size_type Order>
    index_type
    _S_find_free_block () noexcept
    {
        size_type max_order = this->_S_max_order ();
        return this->_S_do_find_free_block<0, max_order + 1, Order> ();
    }
    index_type
    _S_find_free_block (size_type order) noexcept;
    index_type
    _S_do_find_free_block (size_type start, size_type end, size_type order,
                           index_type block_index) noexcept;
    template <size_type BlockIndex, size_type CurrentOrder,
              size_type TargetOrder>
    void
    _S_do_split_block () noexcept
    {
        if (CurrentOrder > TargetOrder)
            {
                _S_BlockOwner &block =
                    this->_M_buddy._M_block_storage[BlockIndex];
                const auto new_size = block._S_size () >> 1;
                const size_type new_block_index = this->_M_buddy._M_block_count;
                ++this->_M_buddy._M_block_count;
                _S_BlockOwner &buddy =
                    this->_M_buddy._M_block_storage[new_block_index];
                auto split_point =
                    block._M_begin + (new_size / sizeof (value_type));
                buddy._S_initialize (split_point, block._M_end);
                block._M_end = split_point;
                this->_S_add_to_free_list<CurrentOrder - 1> (new_block_index);
                this->_S_do_split_block<BlockIndex, CurrentOrder - 1,
                                        TargetOrder> ();
            }
    }

    template <size_type BlockIndex, size_type CurrentOrder,
              size_type TargetOrder>
    void
    _S_split_block () noexcept
    {
        static_assert (BlockIndex < _S_max_blocks,
                       "zzs::base::BuddyAllocator index out of range");
        static_assert (
            TargetOrder <= this->_S_max_order (),
            "zzs::base::BuddyAllocator target order exceeds max order");
        if (!this->_M_buddy._M_block_storage[BlockIndex]._S_is_valid ())
            return;
        else
            this->_S_do_split_block<BlockIndex, CurrentOrder, TargetOrder> ();
    }
    void
    _S_split_block (index_type block_index, size_type current_order,
                    size_type target_order) noexcept;
    void
    _S_do_split_block (index_type block_index, size_type current_order,
                       size_type target_order) noexcept;
    template <size_type Start, size_type End, size_type Order,
              size_type BlockIndex>
    void
    _S_do_coalesce_block () noexcept
    {
        if (Start < End)
            {
                _S_BlockOwner &block =
                    this->_M_buddy._M_block_storage[BlockIndex];
                if (!this->_S_is_valid_block (&block))
                    return;
                const uintptr_t block_addr = reinterpret_cast<uintptr_t> (
                    static_cast<size_type> (block._M_begin));
                const size_type block_size = block._S_size ();
                if (block_size == 0)
                    return;
                const uintptr_t buddy_addr = block_addr ^ block_size;
                const index_type buddy_idx =
                    this->_S_find_block_pointer (buddy_addr);
                if (buddy_idx == _S_BuddyState::uninit)
                    return;
                if (_S_BlockOwner &buddy =
                        this->_M_buddy._M_block_storage[buddy_idx];
                    buddy._S_size () != block_size)
                    return;
                this->_S_remove_from_free_list<buddy_idx, Order> ();
                const size_type new_size = block_size << 1;
                block._M_end =
                    block._M_begin
                    + static_cast<index_type> (new_size / sizeof (value_type));
                this->_S_do_coalesce_block<Start + 1, End, Order + 1,
                                           BlockIndex> ();
            }
    }
    template <size_type BlockIndex, size_type Order>
    void
    _S_coalesce_block () noexcept
    {
        static_assert (BlockIndex < _S_max_blocks,
                       "zzs::base::BuddyAllocator index out of range");
        size_type max_order = this->_S_max_order ();
        this->_S_do_coalesce_block<0, max_order + 1, Order, BlockIndex> ();
    }
    void
    _S_coalesce_block (const index_type block_index) noexcept
    {
        if (block_index >= _S_max_blocks)
            return;
        size_type max_order = this->_S_max_order ();
        for (size_type order = 0; order <= max_order; ++order)
            {
                _S_BlockOwner &block =
                    this->_M_buddy._M_block_storage[block_index];
                if (!this->_S_is_valid_block (&block))
                    return;
                const uintptr_t block_addr = reinterpret_cast<uintptr_t> (
                    static_cast<size_type> (block._M_begin));
                const size_type block_size = block._S_size ();
                if (block_size == 0)
                    return;
                const uintptr_t buddy_addr = block_addr ^ block_size;
                const index_type buddy_idx =
                    this->_S_find_block_pointer (buddy_addr);
                if (buddy_idx == _S_BuddyState::uninit)
                    return;
                if (_S_BlockOwner &buddy =
                        this->_M_buddy._M_block_storage[buddy_idx];
                    buddy._S_size () != block_size)
                    return;
                this->_S_remove_from_free_list (buddy_idx, order);
                const size_type new_size = block_size << 1;
                block._M_end =
                    block._M_begin
                    + static_cast<index_type> (new_size / sizeof (value_type));
            }
    }
    template <size_type BlockIndex, size_type Order>
    void
    _S_add_to_free_list () noexcept
    {
        _S_BlockOwner &block = this->_M_buddy._M_block_storage[BlockIndex];
        static_assert (!std::is_reference_v<_S_BlockOwner>,
                       "zzs::base::BuddyAllocator invalid block state");
        size_type free_list_index = _S_max_order () - Order;
        block._M_next_block = this->_M_buddy._M_free_heads[free_list_index];
        this->_M_buddy._M_free_heads[free_list_index] = BlockIndex;
    }
    template <size_type Order>
    void
    _S_add_to_free_list (const size_type block_index) noexcept
    {
        _S_BlockOwner &block = this->_M_buddy._M_block_storage[block_index];
        static_assert (!std::is_reference_v<_S_BlockOwner>,
                       "zzs::base::BuddyAllocator invalid block state");
        size_type free_list_index = this->_S_max_order () - Order;
        block._M_next_block = this->_M_buddy._M_free_heads[free_list_index];
        this->_M_buddy._M_free_heads[free_list_index] = block_index;
    }
    size_type
    _S_find_block_pointer (uintptr_t block_ptr) const noexcept;
    template <size_type BlockIndex, size_type Order>
    void
    _S_remove_from_free_list () const noexcept
    {
        _S_BlockOwner &block = this->_M_buddy._M_block_storage[BlockIndex];
        constexpr size_type free_list_index = _S_max_order () - Order;
        if (this->_M_buddy._M_free_heads[free_list_index] == BlockIndex)
            this->_M_buddy._M_free_heads[free_list_index] = block._M_next_block;
        else
            {
                index_type prev = this->_M_buddy._M_free_heads[free_list_index];
                while (prev != _S_BuddyState::uninit
                       && this->_M_buddy._M_block_storage[prev]._M_next_block
                              != BlockIndex)
                    prev = this->_M_buddy._M_block_storage[prev]._M_next_block;
                if (prev != _S_BuddyState::uninit)
                    this->_M_buddy._M_block_storage[prev]._M_next_block =
                        block._M_next_block;
            }
    }
    void
    _S_remove_from_free_list (index_type block_index,
                              const size_type order) noexcept;
    void
    _S_add_to_free_list (index_type block_index,
                         const size_type order) noexcept;
};

template <class T, unsigned int Sz> class TlsfAllocator
{
  public:
    static_assert (
        !std::is_reference_v<T>,
        "zzs::base::TlsfAllocator required a non-reference value type");
    static_assert (!std::is_void_v<T>,
                   "zzs::base::TlsfAllocator cannot allocate void");
    static_assert (
        std::is_destructible_v<T>,
        "zzs::base::TlsfAllocator required a destructible value type");
    static_assert (std::is_object_v<T>,
                   "zzs::base::TlsfAllocator required an object type");
    static_assert (std::is_trivially_copyable_v<T>,
                   "zzs::base::TlsfAllocator required a trivially "
                   "copyable value type");
    static_assert (Sz > 0,
                   "zzs::base::TlsfAllocator capacity must be valid");

    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using iterator = BasicIterator<value_type>;
    using const_iterator = const BasicIterator<value_type>;
    using storage_type = StoreBase<T, Sz>;
    static constexpr size_type capacity_value = Sz;

    TlsfAllocator &
    operator= (const TlsfAllocator &) noexcept = delete;
    TlsfAllocator (const TlsfAllocator &other) noexcept = default;
    TlsfAllocator (TlsfAllocator &&other) noexcept = delete;
    TlsfAllocator &
    operator= (TlsfAllocator &&other) noexcept = delete;
    TlsfAllocator () noexcept;

    template <unsigned int N>
        requires (N > 0 && N <= Sz)
    [[nodiscard]] Maybe<pointer, AllocError>
    allocate () noexcept;
    [[nodiscard]] Maybe<pointer, AllocError>
    allocate (size_type count) noexcept;
    void
    deallocate (pointer p) noexcept;
    ~TlsfAllocator () noexcept = default;
    iterator
    begin () const noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    const_iterator
    begin () noexcept
    {
        return iterator (this->_M_storage.data ());
    }
    iterator
    end () noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    const_iterator
    end () const noexcept
    {
        return iterator (this->_M_storage.data () + Sz);
    }
    storage_type
    raw () const noexcept
    {
        return this->_M_storage;
    }
    const_iterator
    cbegin () const noexcept
    {
        return static_cast<const_iterator> (this->begin ());
    }
    const_iterator
    cend () const noexcept
    {
        return static_cast<const_iterator> (this->end ());
    }
    [[nodiscard]] size_type
    max_size () const noexcept
    {
        return static_cast<size_type> (-1) / sizeof (value_type);
    }
    [[nodiscard]] size_type
    used () const noexcept
    {
        return this->_M_used_bytes;
    }
    [[nodiscard]] size_type
    capacity () const noexcept
    {
        return static_cast<size_type> (Sz) * sizeof (value_type);
    }
    [[nodiscard]] bool
    is_empty () const noexcept
    {
        return this->_M_used_bytes == 0;
    }
    void
    reset () const noexcept;

  private:
    static constexpr size_type _S_sl_bits = 2;
    static constexpr size_type _S_sl_count = size_type{1} << _S_sl_bits;
    static constexpr size_type _S_sl_mask = _S_sl_count - 1;
    // Two low bits of every block size carry the flags below, so the block
    // alignment is never smaller than 4 whatever alignof(T) is.
    static constexpr size_type _S_min_align = 4;
    static constexpr size_type
    _S_alignment_helper () noexcept
    {
        const size_type natural = alignof (T) > alignof (size_type)
                                      ? alignof (T)
                                      : alignof (size_type);
        return natural > _S_min_align ? natural : _S_min_align;
    }
    static constexpr size_type _S_alignment = _S_alignment_helper ();

    struct block_header
    {
        size_type _M_size; // total block size in bytes; bit 0: this block is
                           // free, bit 1: previous block is free
        size_type _M_prev_size; // size of the previous physical block; valid
                                // only while bit 1 is set
        size_type _M_next_free; // offset of the next free block in the size
                                // class, _S_none at list tail
        size_type _M_prev_free; // offset of the previous free block in the size
                                // class, _S_none at list head
    };

    static constexpr size_type _S_free_bit = 1;
    static constexpr size_type _S_prev_free_bit = 2;
    static constexpr size_type _S_flag_mask = 3;
    static constexpr size_type _S_none = static_cast<size_type> (-1);

    static constexpr size_type
    _S_raw_bytes () noexcept
    {
        return static_cast<size_type> (Sz) * sizeof (T);
    }
    static constexpr size_type
    _S_align_up (const size_type value) noexcept
    {
        return (value + _S_alignment - 1) & ~(_S_alignment - 1);
    }
    static constexpr size_type
    _S_header_bytes () noexcept
    {
        return _S_align_up (sizeof (block_header));
    }
    static constexpr size_type
    _S_min_block () noexcept
    {
        return _S_header_bytes ();
    }
    static constexpr size_type
    _S_usable_bytes () noexcept
    {
        return _S_raw_bytes () & ~(_S_alignment - 1);
    }
    static constexpr size_type
    _S_msb (const size_type value) noexcept
    {
        size_type index = 0;
        size_type bits = value;
        while (bits >>= 1)
            ++index;
        return index;
    }
    static constexpr size_type _S_fl_min = _S_msb (_S_min_block ());
    static constexpr size_type _S_fl_count =
        _S_usable_bytes () >= _S_min_block ()
            ? _S_msb (_S_usable_bytes ()) - _S_fl_min + 1
            : 1;

    static_assert (
        static_cast<size_type> (Sz) <= static_cast<size_type> (-1) / sizeof (T),
        "zzs::base::TlsfAllocator pool size overflows size_type");
    static_assert (_S_min_block () >= (size_type{1} << _S_sl_bits),
                   "zzs::base::TlsfAllocator minimum block must cover "
                   "one second-level shift");
    static_assert (_S_raw_bytes ()
                       <= static_cast<size_type> (-1) - _S_min_block (),
                   "zzs::base::TlsfAllocator pool size leaves no room "
                   "for size arithmetic");
    static_assert (_S_fl_count <= 64, "zzs::base::TlsfAllocator pool "
                                      "needs more than 64 first-level classes");

    static size_type
    _S_fl_index (const size_type size) noexcept
    {
        return _S_msb (size) - _S_fl_min;
    }
    static size_type
    _S_sl_index (const size_type size) noexcept
    {
        return (size >> (_S_msb (size) - _S_sl_bits)) & _S_sl_mask;
    }
    // Round a request up to the width of its second-level class, so every
    // block reachable from the located free lists is guaranteed to fit and
    // the search stays O(1). Costs at most one class width of reservation.
    static size_type
    _S_round_class (const size_type size) noexcept
    {
        const size_type width = size_type{1} << (_S_msb (size) - _S_sl_bits);
        return (size + width - 1) & ~(width - 1);
    }
    static block_header *
    _S_header_at (void *base, const size_type offset) noexcept
    {
        // Every offset is a multiple of _S_alignment, so the reconstructed
        // header pointer is correctly aligned and stays inside the pool.
        return static_cast<block_header *> (
            static_cast<void *> (static_cast<unsigned char *> (base) + offset));
    }
    static block_header *
    _S_emplace_header (void *base, const size_type offset) noexcept
    {
        // block_header is trivially copyable: placement-new starts its
        // lifetime in pool bytes that otherwise hold T elements; the pool
        // is treated as a raw byte arena from the allocator's viewpoint.
        return ::new (static_cast<void *> (static_cast<unsigned char *> (base)
                                           + offset)) block_header ();
    }

    void *
    _S_base () noexcept
    {
        return static_cast<void *> (this->_M_storage.data ());
    }
    void
    _S_construct_pool () noexcept;
    void
    _S_insert_free (const size_type offset) noexcept;
    void
    _S_remove_free (const size_type offset) noexcept;
    size_type
    _S_locate (const size_type size) noexcept;

    alignas (_S_alignment) storage_type _M_storage{};
    size_type _M_free_heads[_S_fl_count][_S_sl_count];
    uint64_t _M_sl_bitmap[_S_fl_count]{};
    uint64_t _M_fl_bitmap = 0;
    size_type _M_used_bytes = 0;
};

template <typename T, int Sz> StoreBase<T, Sz>::StoreBase () noexcept = default;

template <typename T, int Sz>
StoreBase<T, Sz>::~StoreBase () noexcept = default;

template <typename T> StoreBaseSingle<T>::StoreBaseSingle () noexcept = default;

template <typename T>
StoreBaseSingle<T>::~StoreBaseSingle () noexcept = default;

template <class T, unsigned int Sz>
template <unsigned int N>
    requires (N > 0 && N <= Sz)
Maybe<T *, AllocError>
BumpAllocator<T, Sz>::allocate () noexcept
{
    static_assert (
        N <= Sz, "zzs::base::BumpAllocator size cannot exceed capacity");
    using result_type = Maybe<pointer, AllocError>;
    const size_type offset = this->_M_offset;
    this->_M_offset += N;
    if (offset + N > Sz)
        return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
    return this->_M_storage.data () + offset;
}
template <class T, unsigned int Sz>
Maybe<T *, AllocError>
BumpAllocator<T, Sz>::allocate (const size_type count) noexcept
{
    using result_type = Maybe<pointer, AllocError>;
    if (count == 0)
        return result_type (AllocError (AllocError::Kind::ZERO_SIZE));
    if (count > static_cast<size_type> (Sz - this->_M_offset))
        {
            return result_type (AllocError (AllocError::Kind::TOO_LARGE));
        }
    const size_type offset = this->_M_offset;
    this->_M_offset += count;
    return this->_M_storage.data () + offset;
}
template <class T, unsigned int Sz>
void BumpAllocator<T, Sz>::deallocate (pointer) noexcept {};

template <class T, unsigned int Sz>
BumpAllocator<T, Sz>::~BumpAllocator () noexcept = default;

template <class T, unsigned int Sz>
void
BumpAllocator<T, Sz>::reset () const
{
    this->~BumpAllocator ();
    ::new (static_cast<void *> (const_cast<BumpAllocator *> (this)))
        BumpAllocator ();
};
template <class T, unsigned int Sz>
template <unsigned int N>
Maybe<T *, AllocError>
BitmapAllocator<T, Sz>::allocate () noexcept
{
    static_assert (N > 0 && N < Sz,
                   "zzs::base::BitmapAllocator invalid size");
    using result_type = Maybe<pointer, AllocError>;
    for (size_type word_idx = 0; word_idx < bitmap_words; ++word_idx)
        {
            uint64_t word = this->_M_allocation_bitmap[word_idx];
            if (word == UINT64_MAX)
                continue;
            const size_type start_bit = __builtin_ctzll (~word);
            size_type bits_needed = N;
            const size_type available = 64 - start_bit;
            if (available >= bits_needed)
                {
                    uint64_t mask = 0;
                    if (bits_needed >= 64)
                        mask = UINT64_MAX << start_bit;
                    else
                        mask = ((UINT64_C (1) << bits_needed) - 1) << start_bit;
                    this->_M_allocation_bitmap[word_idx] |= mask;
                    this->_M_bitmap_count += N;
                    size_type element_idx = word_idx * 64 + start_bit;
                    return this->_M_storage.data () + element_idx;
                }
            size_type consecutive = available;
            for (size_type next_word = word_idx + 1;
                 next_word < bitmap_words && consecutive < bits_needed;
                 ++next_word)
                {
                    const uint64_t next_word_val =
                        this->_M_allocation_bitmap[next_word];
                    size_type next_bits = __builtin_ctzll (~next_word_val);
                    if (next_bits == 0)
                        {
                            consecutive += 64;
                            if (consecutive >= bits_needed)
                                {
                                    return this->_S_handle_needed<N> (
                                        word_idx, next_word, start_bit,
                                        bits_needed, available);
                                }
                            continue;
                        }
                    consecutive += next_bits;
                    if (consecutive >= bits_needed)
                        return this->_S_handle_needed<N> (
                            word_idx, next_word, start_bit, bits_needed,
                            available);
                }
        }
    return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
}
template <class T, unsigned int Sz>
void
BitmapAllocator<T, Sz>::deallocate (pointer ptr) noexcept
{
    const uintptr_t ptr_addr = reinterpret_cast<uintptr_t> (ptr);
    const uintptr_t base_addr =
        reinterpret_cast<uintptr_t> (this->_M_storage.data ());
    if (ptr_addr < base_addr)
        return;
    const size_type element_idx = (ptr_addr - base_addr) / sizeof (value_type);
    if (element_idx >= max_bitmaps)
        return;
    const size_type word_idx = element_idx >> 6;
    const size_type bit_idx = element_idx & 63;
    const uint64_t word = this->_M_allocation_bitmap[word_idx];
    if ((word & (UINT64_C (1) << bit_idx)) == 0)
        return;
    size_type count = 1;
    while (bit_idx + count < 64
           && (word & (UINT64_C (1) << (bit_idx + count))) != 0)
        ++count;
    if (bit_idx + count == 64)
        {
            for (size_type w = word_idx + 1; w < bitmap_words; ++w)
                {
                    const uint64_t next_word = this->_M_allocation_bitmap[w];
                    size_type next_count = 0;
                    while (next_count < 64
                           && (next_word & (UINT64_C (1) << next_count)) != 0)
                        ++next_count;
                    count += next_count;
                    if (next_count < 64)
                        break;
                }
        }
    const uint64_t mask =
        (count == 64) ? UINT64_MAX : ((UINT64_C (1) << count) - 1) << bit_idx;
    this->_M_allocation_bitmap[word_idx] &= ~mask;
    if (bit_idx + count > 64)
        {
            size_type remaining = bit_idx + count - 64;
            for (size_type w = word_idx + 1; w < bitmap_words && remaining > 0;
                 ++w)
                {
                    const uint64_t next_mask =
                        (remaining >= 64) ? UINT64_MAX
                                          : ((UINT64_C (1) << remaining) - 1);
                    this->_M_allocation_bitmap[w] &= ~next_mask;
                    if (remaining > 64)
                        remaining -= 64;
                    else
                        remaining = 0;
                }
        }
    this->_M_bitmap_count -= count;
}

template <class T, unsigned int Sz>
template <unsigned int N>
BitmapAllocator<T, Sz>::pointer
BitmapAllocator<T, Sz>::_S_handle_needed (const size_type word_idx,
                                          const size_type next_word,
                                          const size_type start_bit,
                                          const size_type bits_needed,
                                          const size_type available) noexcept
{
    uint64_t first_mask = UINT64_MAX << start_bit;
    this->_M_allocation_bitmap[word_idx] |= first_mask;
    for (size_type w = word_idx + 1; w < next_word; ++w)
        this->_M_allocation_bitmap[w] = UINT64_MAX;
    const size_type remaining_bits = bits_needed - available;
    uint64_t last_mask = 0;
    if (remaining_bits >= 64)
        last_mask = UINT64_MAX;
    else
        last_mask = (UINT64_C (1) << remaining_bits) - 1;
    this->_M_allocation_bitmap[next_word] |= last_mask;
    this->_M_bitmap_count += N;
    size_type element_idx = (word_idx << 6) + start_bit;
    return this->_M_storage.data () + element_idx;
}
template <class T, unsigned int Sz>
BuddyAllocator<T, Sz>::BuddyAllocator () noexcept
{
    this->_S_construct_allocator ();
}

template <class T, unsigned int Sz>
template <unsigned int N>
Maybe<T *, AllocError>
BuddyAllocator<T, Sz>::allocate () noexcept
{
    static_assert (N <= Sz, "zzs::base::BuddyAllocator invalid size");
    return this->allocate (static_cast<size_type> (N));
}

template <class T, unsigned int Sz>
Maybe<T *, AllocError>
BuddyAllocator<T, Sz>::allocate (const size_type count) noexcept
{
    using result_type = Maybe<pointer, AllocError>;
    if (count == 0)
        return result_type (AllocError (AllocError::Kind::ZERO_SIZE));
    if (count > Sz)
        return result_type (AllocError (AllocError::Kind::TOO_LARGE));
    const size_type min_block_size = _S_min_block_size ();
    size_type max_order = this->_S_max_order ();
    size_type order = 0;
    size_type block_size = min_block_size;
    while (block_size < count * sizeof (value_type) && order < max_order)
        {
            block_size <<= 1;
            ++order;
        }
    if (order > max_order)
        return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
    index_type block_idx = this->_S_find_free_block (order);
    if (block_idx == _S_BuddyState::uninit)
        return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
    _S_BlockOwner &block = this->_M_buddy._M_block_storage[block_idx];
    if (!_S_is_valid_block (&block))
        return result_type (AllocError (AllocError::Kind::INVALID_BLOCK));
    this->_M_used_bytes += block._S_size ();
    this->_S_remove_from_free_list (block_idx, order);
    return result_type (this->_M_storage.data ()
                        + static_cast<size_type> (block._M_begin));
}

template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::deallocate (pointer p) noexcept
{
    if (p == nullptr)
        return;
    const uintptr_t ptr_addr = reinterpret_cast<uintptr_t> (p);
    const uintptr_t base_addr =
        reinterpret_cast<uintptr_t> (this->_M_storage.data ());
    for (size_type i = 0; i < this->_M_buddy._M_block_count; ++i)
        {
            _S_BlockOwner &block = this->_M_buddy._M_block_storage[i];
            if (!block._S_is_allocated ())
                continue;
            const uintptr_t block_start =
                base_addr + (block._M_begin * sizeof (value_type));
            const uintptr_t block_end =
                base_addr + (block._M_end * sizeof (value_type));
            if (ptr_addr >= block_start && ptr_addr < block_end)
                {
                    const size_type block_size = block._S_size ();
                    block._S_mark_deallocated ();
                    this->_M_used_bytes -= block_size;
                    this->_S_coalesce_block (i);
                    return;
                }
        }
}

template <class T, unsigned int Sz>
BuddyAllocator<T, Sz>::~BuddyAllocator () noexcept = default;

template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::reset () const
{
    this->~BuddyAllocator ();
    ::new (static_cast<void *> (const_cast<BuddyAllocator *> (this)))
        BuddyAllocator ();
}
template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::_S_construct_allocator () noexcept
{
    size_type initial_idx = 0;
    this->_M_buddy._M_block_count = initial_idx + 1;
    _S_BlockOwner &initial_block = this->_M_buddy._M_block_storage[initial_idx];
    initial_block._S_initialize (0, Sz);
    initial_block._M_next_block = _S_BuddyState::uninit;
    this->_S_add_to_free_list<0, _S_max_order ()> ();
}
template <class T, unsigned int Sz>
bool
BuddyAllocator<T, Sz>::_S_is_valid_block (const _S_BlockOwner *block) noexcept
{
    if (block == nullptr)
        return false;
    else
        return block->_S_is_valid ();
}
template <class T, unsigned int Sz>
bool
BuddyAllocator<T, Sz>::_S_is_block_allocated (
    const _S_BlockOwner *block) noexcept
{
    if (!_S_is_valid_block (block))
        return false;
    else
        return block->_S_is_allocated ();
}
template <class T, unsigned int Sz>
BuddyAllocator<T, Sz>::index_type
BuddyAllocator<T, Sz>::_S_find_free_block (const size_type order) noexcept
{
    size_type max_order = this->_S_max_order ();
    return this->_S_do_find_free_block (0, max_order + 1, order, 0);
}
template <class T, unsigned int Sz>
BuddyAllocator<T, Sz>::index_type
BuddyAllocator<T, Sz>::_S_do_find_free_block (
    size_type start, const size_type end, const size_type order,
    const index_type block_index) noexcept
{
    if (start < end)
        {
            size_type curr_order = end - 1 - start;
            if (const index_type free_head =
                    this->_M_buddy._M_free_heads[start];
                free_head == block_index)
                {
                    this->_S_remove_from_free_list (block_index, curr_order);
                    if (curr_order > order)
                        this->_S_split_block (block_index, curr_order, order);
                    return block_index;
                }
            if (block_index + 1 < _S_max_blocks)
                return this->_S_do_find_free_block (start, end, order,
                                                    block_index + 1);
            else
                return this->_S_do_find_free_block (start + 1, end, order, 0);
        }
    return _S_BuddyState::uninit;
}
template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::_S_split_block (index_type block_index,
                                       size_type current_order,
                                       size_type target_order) noexcept
{
    if (block_index >= _S_max_blocks || target_order > this->_S_max_order ())
        return;
    if (!this->_M_buddy._M_block_storage[block_index]._S_is_valid ())
        return;
    this->_S_do_split_block (block_index, current_order, target_order);
}
template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::_S_do_split_block (index_type block_index,
                                          size_type current_order,
                                          size_type target_order) noexcept
{
    while (current_order > target_order)
        {
            _S_BlockOwner &block = this->_M_buddy._M_block_storage[block_index];
            const auto new_size = block._S_size () >> 1;
            const index_type new_block_index = this->_M_buddy._M_block_count;
            ++this->_M_buddy._M_block_count;
            _S_BlockOwner &buddy =
                this->_M_buddy._M_block_storage[new_block_index];
            auto split_point =
                block._M_begin
                + static_cast<index_type> (new_size / sizeof (value_type));
            buddy._S_initialize (split_point, block._M_end);
            block._M_end = split_point;
            this->_S_add_to_free_list (new_block_index, current_order - 1);
            current_order--;
        }
}
template <class T, unsigned int Sz>
BuddyAllocator<T, Sz>::size_type
BuddyAllocator<T, Sz>::_S_find_block_pointer (
    const uintptr_t block_ptr) const noexcept
{
    for (size_type i = 0; i < this->_M_buddy._M_block_count; ++i)
        {
            const _S_BlockOwner &block = this->_M_buddy._M_block_storage[i];
            if (!block._S_is_allocated ())
                continue;
            if (const uintptr_t block_start = reinterpret_cast<uintptr_t> (
                    static_cast<size_type> (block._M_begin));
                block_ptr == block_start)
                return i;
        }
    return _S_BuddyState::uninit;
}
template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::_S_remove_from_free_list (index_type block_index,
                                                 const size_type order) noexcept
{
    _S_BlockOwner &block = this->_M_buddy._M_block_storage[block_index];
    const size_type free_list_index = _S_max_order () - order;
    if (this->_M_buddy._M_free_heads[free_list_index] == block_index)
        this->_M_buddy._M_free_heads[free_list_index] = block._M_next_block;
    else
        {
            index_type prev = this->_M_buddy._M_free_heads[free_list_index];
            while (prev != _S_BuddyState::uninit
                   && this->_M_buddy._M_block_storage[prev]._M_next_block
                          != block_index)
                prev = this->_M_buddy._M_block_storage[prev]._M_next_block;
            if (prev != _S_BuddyState::uninit)
                this->_M_buddy._M_block_storage[prev]._M_next_block =
                    block._M_next_block;
        }
}
template <class T, unsigned int Sz>
void
BuddyAllocator<T, Sz>::_S_add_to_free_list (index_type block_index,
                                            const size_type order) noexcept
{
    _S_BlockOwner &block = this->_M_buddy._M_block_storage[block_index];
    const size_type free_list_index = _S_max_order () - order;
    block._M_next_block = this->_M_buddy._M_free_heads[free_list_index];
    this->_M_buddy._M_free_heads[free_list_index] = block_index;
}
template <class T, unsigned int Sz>
TlsfAllocator<T, Sz>::TlsfAllocator () noexcept
{
    this->_S_construct_pool ();
}

template <class T, unsigned int Sz>
void
TlsfAllocator<T, Sz>::_S_construct_pool () noexcept
{
    for (size_type fl = 0; fl < _S_fl_count; ++fl)
        {
            this->_M_sl_bitmap[fl] = 0;
            for (size_type sl = 0; sl < _S_sl_count; ++sl)
                this->_M_free_heads[fl][sl] = _S_none;
        }
    this->_M_fl_bitmap = 0;
    this->_M_used_bytes = 0;
    const size_type usable = _S_usable_bytes ();
    if (usable >= _S_min_block ())
        {
            block_header *const initial =
                _S_emplace_header (this->_S_base (), 0);
            initial->_M_size = usable | _S_free_bit;
            initial->_M_prev_size = 0;
            initial->_M_next_free = _S_none;
            initial->_M_prev_free = _S_none;
            this->_S_insert_free (0);
        }
}

template <class T, unsigned int Sz>
void
TlsfAllocator<T, Sz>::_S_insert_free (const size_type offset) noexcept
{
    block_header *const header = _S_header_at (this->_S_base (), offset);
    const size_type size = header->_M_size & ~_S_flag_mask;
    const size_type fl = _S_fl_index (size);
    const size_type sl = _S_sl_index (size);
    size_type &head = this->_M_free_heads[fl][sl];
    header->_M_prev_free = _S_none;
    header->_M_next_free = head;
    if (head != _S_none)
        _S_header_at (this->_S_base (), head)->_M_prev_free = offset;
    head = offset;
    this->_M_sl_bitmap[fl] |= (uint64_t{1} << sl);
    this->_M_fl_bitmap |= (uint64_t{1} << fl);
}

template <class T, unsigned int Sz>
void
TlsfAllocator<T, Sz>::_S_remove_free (const size_type offset) noexcept
{
    block_header *const header = _S_header_at (this->_S_base (), offset);
    const size_type size = header->_M_size & ~_S_flag_mask;
    const size_type fl = _S_fl_index (size);
    const size_type sl = _S_sl_index (size);
    if (header->_M_prev_free != _S_none)
        _S_header_at (this->_S_base (), header->_M_prev_free)->_M_next_free =
            header->_M_next_free;
    else
        this->_M_free_heads[fl][sl] = header->_M_next_free;
    if (header->_M_next_free != _S_none)
        _S_header_at (this->_S_base (), header->_M_next_free)->_M_prev_free =
            header->_M_prev_free;
    if (this->_M_free_heads[fl][sl] == _S_none)
        {
            this->_M_sl_bitmap[fl] &= ~(uint64_t{1} << sl);
            if (this->_M_sl_bitmap[fl] == 0)
                this->_M_fl_bitmap &= ~(uint64_t{1} << fl);
        }
}

template <class T, unsigned int Sz>
TlsfAllocator<T, Sz>::size_type
TlsfAllocator<T, Sz>::_S_locate (const size_type size) noexcept
{
    const size_type fl = _S_fl_index (size);
    const size_type sl = _S_sl_index (size);
    if (const uint64_t sl_bits = this->_M_sl_bitmap[fl] >> sl; sl_bits != 0)
        return this->_M_free_heads[fl][sl
                                       + static_cast<size_type> (
                                           __builtin_ctzll (sl_bits))];
    if (const uint64_t fl_bits = this->_M_fl_bitmap >> (fl + 1); fl_bits != 0)
        {
            const size_type next_fl =
                fl + 1 + static_cast<size_type> (__builtin_ctzll (fl_bits));
            const size_type next_sl = static_cast<size_type> (
                __builtin_ctzll (this->_M_sl_bitmap[next_fl]));
            return this->_M_free_heads[next_fl][next_sl];
        }
    return _S_none;
}

template <class T, unsigned int Sz>
template <unsigned int N>
    requires (N > 0 && N <= Sz)
Maybe<T *, AllocError>
TlsfAllocator<T, Sz>::allocate () noexcept
{
    return this->allocate (static_cast<size_type> (N));
}

template <class T, unsigned int Sz>
Maybe<T *, AllocError>
TlsfAllocator<T, Sz>::allocate (const size_type count) noexcept
{
    using result_type = Maybe<pointer, AllocError>;
    if (count == 0)
        return result_type (AllocError (AllocError::Kind::ZERO_SIZE));
    const size_type usable = this->_S_usable_bytes ();
    if (count > usable / sizeof (value_type))
        return result_type (AllocError (AllocError::Kind::TOO_LARGE));
    const size_type needed =
        this->_S_align_up (this->_S_min_block () + count * sizeof (value_type));
    if (needed > usable)
        return result_type (AllocError (AllocError::Kind::TOO_LARGE));
    const size_type rounded = this->_S_round_class (needed);
    if (rounded > usable)
        return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
    const size_type block_off = this->_S_locate (rounded);
    if (block_off == _S_none)
        return result_type (AllocError (AllocError::Kind::OUT_OF_MEMORY));
    void *base = this->_S_base ();
    block_header *const header = this->_S_header_at (base, block_off);
    const size_type block_size = header->_M_size & ~_S_flag_mask;
    const size_type prev_flags = header->_M_size & _S_prev_free_bit;
    this->_S_remove_free (block_off);
    size_type granted = block_size;
    const size_type successor_off = block_off + block_size;
    if (block_size - needed >= this->_S_min_block ())
        {
            // Split: keep a header-aligned leading block for the caller and
            // push the tail back onto its size-class free list.
            const size_type remainder_off = block_off + needed;
            const size_type remainder_size = block_size - needed;
            block_header *const remainder =
                this->_S_emplace_header (base, remainder_off);
            remainder->_M_size = remainder_size | _S_free_bit | prev_flags;
            remainder->_M_prev_size = header->_M_prev_size;
            header->_M_size = needed | prev_flags;
            if (successor_off < usable)
                {
                    block_header *const successor =
                        this->_S_header_at (base, successor_off);
                    successor->_M_prev_size = remainder_size;
                    successor->_M_size |= _S_prev_free_bit;
                }
            this->_S_insert_free (remainder_off);
            granted = needed;
        }
    else
        {
            // The block is handed out whole; its successor no longer has a
            // free predecessor, so the previous-free flag must be cleared.
            header->_M_size = block_size | prev_flags;
            if (successor_off < usable)
                this->_S_header_at (base, successor_off)->_M_size &=
                    ~_S_prev_free_bit;
        }
    this->_M_used_bytes += granted;
    const size_type user_offset = block_off + this->_S_header_bytes ();
    return result_type (static_cast<pointer> (static_cast<void *> (
        static_cast<unsigned char *> (base) + user_offset)));
}

template <class T, unsigned int Sz>
void
TlsfAllocator<T, Sz>::deallocate (pointer p) noexcept
{
    // reinterpret_cast: the pointer handed back by allocate() is validated
    // by address range before any dereference, the same convention the
    // allocators above use; the pool is a byte arena, not live T objects.
    const uintptr_t base_addr =
        reinterpret_cast<uintptr_t> (this->_M_storage.data ());
    const uintptr_t user_addr = reinterpret_cast<uintptr_t> (p);
    if (user_addr < base_addr)
        return;
    const size_type user_offset =
        static_cast<size_type> (user_addr - base_addr);
    const size_type usable = this->_S_usable_bytes ();
    if (user_offset < this->_S_header_bytes () || user_offset >= usable)
        return;
    const size_type block_off = user_offset - this->_S_header_bytes ();
    if ((block_off & (_S_alignment - 1)) != 0)
        return;
    void *base = this->_S_base ();
    block_header *const header = this->_S_header_at (base, block_off);
    const size_type size = header->_M_size & ~_S_flag_mask;
    if (size < this->_S_min_block () || (size & (_S_alignment - 1)) != 0
        || size > usable || block_off > usable - size)
        return;
    if ((header->_M_size & _S_free_bit) != 0)
        return; // double free or foreign pointer: precondition violated, drop
                // the request
    // All validation happens before the free lists are touched below.
    const size_type prev_flags = header->_M_size & _S_prev_free_bit;
    size_type prev_size = 0;
    size_type prev_prev_flags = 0;
    size_type prev_prev_size = 0;
    if (prev_flags != 0)
        {
            prev_size = header->_M_prev_size;
            if (prev_size < this->_S_min_block ()
                || (prev_size & (_S_alignment - 1)) != 0
                || prev_size > block_off)
                return;
            const block_header *const prev =
                this->_S_header_at (base, block_off - prev_size);
            prev_prev_flags = prev->_M_size & _S_prev_free_bit;
            prev_prev_size = prev->_M_prev_size;
        }
    // Coalesce with the next physical block when it is free.
    size_type merged_off = block_off;
    size_type merged_size = size;
    const size_type successor_off = block_off + size;
    if (successor_off < usable)
        {
            block_header *const successor =
                this->_S_header_at (base, successor_off);
            if ((successor->_M_size & _S_free_bit) != 0)
                {
                    this->_S_remove_free (successor_off);
                    merged_size += successor->_M_size & ~_S_flag_mask;
                }
        }
    // Coalesce with the previous physical block when it is free.
    if (prev_flags != 0)
        {
            merged_off = block_off - prev_size;
            merged_size += prev_size;
            this->_S_remove_free (merged_off);
        }
    block_header *const merged = this->_S_emplace_header (base, merged_off);
    const size_type merged_prev_flags =
        prev_flags != 0 ? prev_prev_flags : prev_flags;
    merged->_M_size = merged_size | _S_free_bit | merged_prev_flags;
    merged->_M_prev_size = prev_flags != 0 ? prev_prev_size : 0;
    if (const size_type merged_successor = merged_off + merged_size;
        merged_successor < usable)
        {
            block_header *const successor =
                this->_S_header_at (base, merged_successor);
            successor->_M_prev_size = merged_size;
            successor->_M_size |= _S_prev_free_bit;
        }
    this->_S_insert_free (merged_off);
    this->_M_used_bytes -= size;
}

template <class T, unsigned int Sz>
void
TlsfAllocator<T, Sz>::reset () const noexcept
{
    this->~TlsfAllocator ();
    ::new (static_cast<void *> (const_cast<TlsfAllocator *> (this)))
        TlsfAllocator ();
}

template <class T, unsigned int Sz>
void
BitmapAllocator<T, Sz>::reset () const noexcept
{
    this->~BitmapAllocator ();
    ::new (static_cast<void *> (const_cast<BitmapAllocator *> (this)))
        BitmapAllocator ();
}
} // namespace zzs::base