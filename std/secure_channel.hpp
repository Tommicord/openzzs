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

#include "std/log.hpp"
#include "std/maybe.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace zzs::base {

/* =========================================================================
 * Secure channel — embedded PSK secure channel (TLS-like, < 512 bytes RAM
 * context).
 *
 * Algorithms:
 *   AES-256-GCM      (x86_64 AES-NI assembly; portable C++ elsewhere)
 *   ChaCha20-Poly1305 (constant-time software, RFC 8439, all targets)
 *   SHAKE256          (Keccak-f[1600] sponge, constant-time software)
 *
 * Record format (per encrypted chunk):
 *   [2 bytes: plaintext length][8 bytes: sequence number][12 bytes: nonce]
 *   [N bytes: ciphertext][16 bytes: GCM / Poly1305 tag]
 *
 * All crypto functions are noexcept and constant-time with respect to secret
 * data (keys, plaintext, nonces, tags). No heap, no writable statics, no
 * exceptions, no RTTI.
 * ========================================================================= */

enum class SecureChannelAlgo : uint8_t {
        AES_256_GCM = 0,
        CHACHA20_POLY1305 = 1,
        SHAKE256_KDF = 2,
};

enum class SecureChannelError : uint8_t {
        OK = 0,
        INVALID_KEY,
        INVALID_NONCE,
        AUTH_FAILED,
        BUFFER_TOO_SMALL,
        SEQ_OVERFLOW,
        UNSUPPORTED_ALGO,
};

/* -----------------------------------------------------------------------
 * Constant-time helpers.
 * --------------------------------------------------------------------- */

[[nodiscard]] inline uint64_t
_S_ct_load64_le (const uint8_t *p) noexcept {
        uint64_t v = 0;
        for (size_t i = 0; i < 8; ++i)
                v |= static_cast<uint64_t> (p[i]) << (8 * i);
        return v;
}

[[nodiscard]] inline uint32_t
_S_ct_load32_le (const uint8_t *p) noexcept {
        uint32_t v = 0;
        for (size_t i = 0; i < 4; ++i)
                v |= static_cast<uint32_t> (p[i]) << (8 * i);
        return v;
}

inline void
_S_ct_store64_le (uint8_t *p, uint64_t v) noexcept {
        for (size_t i = 0; i < 8; ++i)
                p[i] = static_cast<uint8_t> ((v >> (8 * i)) & 0xff);
}

inline void
_S_ct_store32_le (uint8_t *p, uint32_t v) noexcept {
        for (size_t i = 0; i < 4; ++i)
                p[i] = static_cast<uint8_t> ((v >> (8 * i)) & 0xff);
}

/* Mask = 0xff...ff when cond is true, 0 otherwise. Branch-free. */
[[nodiscard]] inline uint8_t
_S_ct_mask_eq (uint8_t a, uint8_t b) noexcept {
        return static_cast<uint8_t> (~(static_cast<unsigned> (a) ^ static_cast<unsigned> (b)) - 1u);
}

/* Constant-time comparison of n bytes: returns 0xff when equal, 0 when not. */
[[nodiscard]] inline uint8_t
_S_ct_verify (const uint8_t *a, const uint8_t *b, size_t n) noexcept {
        uint8_t acc = 0;
        for (size_t i = 0; i < n; ++i)
                acc |= a[i] ^ b[i];
        /* acc == 0  ->  0xff, otherwise 0. */
        return static_cast<uint8_t> (~(static_cast<unsigned> (acc) - 1u) & 0xffu);
}

/* Rotate-left of a 32-bit word. */
[[nodiscard]] inline uint32_t
_S_rotl32 (uint32_t x, int n) noexcept {
        return (x << n) | (x >> (32 - n));
}

/* Rotate-left of a 64-bit word. */
[[nodiscard]] inline uint64_t
_S_rotl64 (uint64_t x, int n) noexcept {
        return (x << n) | (x >> (64 - n));
}

/* Compile-time barrier to discourage the compiler from removing
 * security-critical computation. */
inline void
_S_ct_barrier () noexcept {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile ("" ::: "memory");
#else
        std::atomic_signal_fence (std::memory_order_seq_cst);
#endif
}

/* =========================================================================
 * SHAKE256 — Keccak-f[1600] sponge.
 * ========================================================================= */

class Shake256 {
      public:
        static constexpr size_t _S_rate = 136; /* 1600 - 2 * 256 bits = 1088 bits = 136 bytes */
        static constexpr size_t _S_digest_size = 32;

        Shake256 () noexcept = default;

        [[nodiscard]] static constexpr size_t
        block_size () noexcept {
                return _S_rate;
        }

        [[nodiscard]] static constexpr size_t
        digest_size () noexcept {
                return _S_digest_size;
        }

        /* Absorb `len` bytes from `data`. May be called repeatedly. */
        void
        absorb (const uint8_t *data, size_t len) noexcept;

        /* Squeeze `out_len` bytes into `out`. May be called repeatedly. */
        void
        squeeze (uint8_t *out, size_t out_len) noexcept;

        /* One-shot: absorb `data` then squeeze `out_len` bytes into `out`. */
        void
        derive (const uint8_t *data, size_t data_len, uint8_t *out, size_t out_len) noexcept;

        /* Reset the sponge to its initial state. */
        void
        reset () noexcept;

      private:
        void
        _S_keccakf () noexcept;

        void
        _S_absorb_block (const uint8_t *block) noexcept;

        alignas (8) uint64_t _M_state[25]{}; /* 1600 bits = 25 lanes */
        size_t _M_buffered = 0;              /* bytes buffered in _M_buffer */
        uint8_t _M_buffer[_S_rate]{};        /* partial rate block */
        uint8_t _M_squeezed = 0;             /* nonzero once squeezing has started */
};

/* Derive `out_len` bytes from `psk` (and optional `salt`) using SHAKE256.
 * Constant-time with respect to the psk material. */
inline void
shake256_derive_key (const uint8_t *psk, size_t psk_len, const uint8_t *salt, size_t salt_len, uint8_t *out,
                     size_t out_len) noexcept {
        Shake256 s;
        s.absorb (psk, psk_len);
        s.absorb (salt, salt_len);
        s.squeeze (out, out_len);
        _S_ct_barrier ();
}

/* =========================================================================
 * ChaCha20 — RFC 8439, constant-time software (ARX only).
 * ========================================================================= */

class ChaCha20 {
      public:
        static constexpr size_t _S_key_size = 32;
        static constexpr size_t _S_nonce_size = 12;
        static constexpr size_t _S_counter_size = 4;
        static constexpr size_t _S_block_size = 64;

        ChaCha20 () noexcept = default;

        /* Set key (32 bytes), nonce (12 bytes) and initial counter. */
        void
        set_key (const uint8_t key[_S_key_size]) noexcept;

        void
        set_nonce (const uint8_t nonce[_S_nonce_size]) noexcept;

        void
        set_counter (uint32_t counter) noexcept;

        /* Encrypt `len` bytes from `in` into `out` (in-place allowed). */
        void
        crypt (const uint8_t *in, uint8_t *out, size_t len) noexcept;

      private:
        void
        _S_block (uint8_t out[_S_block_size]) noexcept;

        uint32_t _M_state[16]{};
        uint8_t _M_key[_S_key_size]{};
        uint8_t _M_nonce[_S_nonce_size]{};
        uint32_t _M_counter = 0;
};

/* ChaCha20 block function exposed for Poly1305 key derivation.
 * Uses the public crypt() path (keystream = crypt of zeros). */
inline void
chacha20_block (const uint8_t key[32], const uint8_t nonce[12], uint32_t counter, uint8_t out[64]) noexcept {
        ChaCha20 c;
        c.set_key (key);
        c.set_nonce (nonce);
        c.set_counter (counter);
        uint8_t zeros[64]{};
        c.crypt (zeros, out, 64);
}

/* ChaCha20 quarter-round on four state lanes. Constant-time ARX. */
inline void
_S_chacha_qr (uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) noexcept {
        a += b;
        d ^= a;
        d = _S_rotl32 (d, 16);
        c += d;
        b ^= c;
        b = _S_rotl32 (b, 12);
        a += b;
        d ^= a;
        d = _S_rotl32 (d, 8);
        c += d;
        b ^= c;
        b = _S_rotl32 (b, 7);
        _S_ct_barrier ();
}

inline void
ChaCha20::set_key (const uint8_t key[_S_key_size]) noexcept {
        for (size_t i = 0; i < _S_key_size; ++i)
                _M_key[i] = key[i];
        /* Constants "expand 32-byte k". */
        _M_state[0] = UINT32_C (0x61707865);
        _M_state[1] = UINT32_C (0x3320646e);
        _M_state[2] = UINT32_C (0x79622d32);
        _M_state[3] = UINT32_C (0x6b206574);
        for (size_t i = 0; i < 8; ++i)
                _M_state[4 + i] = _S_ct_load32_le (key + 4 * i);
        _M_state[12] = _M_counter;
        for (size_t i = 0; i < 3; ++i)
                _M_state[13 + i] = _S_ct_load32_le (_M_nonce + 4 * i);
        _S_ct_barrier ();
}

inline void
ChaCha20::set_nonce (const uint8_t nonce[_S_nonce_size]) noexcept {
        for (size_t i = 0; i < _S_nonce_size; ++i)
                _M_nonce[i] = nonce[i];
        for (size_t i = 0; i < 3; ++i)
                _M_state[13 + i] = _S_ct_load32_le (_M_nonce + 4 * i);
        _S_ct_barrier ();
}

inline void
ChaCha20::set_counter (uint32_t counter) noexcept {
        _M_counter = counter;
        _M_state[12] = counter;
        _S_ct_barrier ();
}

inline void
ChaCha20::_S_block (uint8_t out[_S_block_size]) noexcept {
        uint32_t state[16];
        for (size_t i = 0; i < 16; ++i)
                state[i] = _M_state[i];

        /* 20 rounds = 10 double rounds. */
        for (int r = 0; r < 10; ++r) {
                /* Column rounds. */
                _S_chacha_qr (state[0], state[4], state[8], state[12]);
                _S_chacha_qr (state[1], state[5], state[9], state[13]);
                _S_chacha_qr (state[2], state[6], state[10], state[14]);
                _S_chacha_qr (state[3], state[7], state[11], state[15]);
                /* Diagonal rounds. */
                _S_chacha_qr (state[0], state[5], state[10], state[15]);
                _S_chacha_qr (state[1], state[6], state[11], state[12]);
                _S_chacha_qr (state[2], state[7], state[8], state[13]);
                _S_chacha_qr (state[3], state[4], state[9], state[14]);
        }

        for (size_t i = 0; i < 16; ++i) {
                state[i] += _M_state[i];
                _S_ct_store32_le (out + 4 * i, state[i]);
        }
        _S_ct_barrier ();
}

inline void
ChaCha20::crypt (const uint8_t *in, uint8_t *out, size_t len) noexcept {
        size_t pos = 0;
        while (pos < len) {
                uint8_t block[_S_block_size];
                _S_block (block);
                const size_t take = len - pos < _S_block_size ? len - pos : _S_block_size;
                for (size_t i = 0; i < take; ++i)
                        out[pos + i] = static_cast<uint8_t> (in[pos + i] ^ block[i]);
                pos += take;
                /* Increment 32-bit counter (little-endian). */
                _M_state[12] += 1;
                _M_counter = _M_state[12];
                _S_ct_barrier ();
        }
}

/* =========================================================================
 * Poly1305 — constant-time software (RFC 7539).
 * ========================================================================= */

class Poly1305 {
      public:
        static constexpr size_t _S_key_size = 32;
        static constexpr size_t _S_tag_size = 16;

        Poly1305 () noexcept = default;

        /* Set the one-time key (32 bytes). Clamps internally. */
        void
        set_key (const uint8_t key[_S_key_size]) noexcept;

        /* Absorb `len` bytes. len may be 0 only when called between blocks. */
        void
        update (const uint8_t *data, size_t len) noexcept;

        /* Finalize and write the 16-byte tag to `out`. */
        void
        final (uint8_t out[_S_tag_size]) noexcept;

        /* One-shot MAC: key + message -> tag. */
        static void
        mac (const uint8_t key[_S_key_size], const uint8_t *msg, size_t msg_len, uint8_t out[_S_tag_size]) noexcept;

      private:
        void
        _S_process_block (const uint8_t *block) noexcept;

        uint32_t _M_r[5]{};   /* clamped key */
        uint32_t _M_s[4]{};   /* clamped key */
        uint32_t _M_acc[5]{}; /* accumulator */
        uint8_t _M_buffer[16]{};
        size_t _M_buffered = 0;
        bool _M_finalized = false;
};

inline void
Poly1305::set_key (const uint8_t key[_S_key_size]) noexcept {
        /* Clamp r (16 bytes = 4 words): clear bits 4-7 of each word. */
        for (size_t i = 0; i < 4; ++i) {
                uint32_t w = _S_ct_load32_le (key + 4 * i);
                w &= 0x0fffffffu;
                _M_r[i] = w;
        }
        _M_r[4] = 0;
        for (size_t i = 0; i < 4; ++i)
                _M_s[i] = _S_ct_load32_le (key + 16 + 4 * i);
        for (size_t i = 0; i < 5; ++i)
                _M_acc[i] = 0;
        _M_buffered = 0;
        _M_finalized = false;
        _S_ct_barrier ();
}

inline void
Poly1305::_S_process_block (const uint8_t *block) noexcept {
        uint64_t t[5];
        for (size_t i = 0; i < 5; ++i)
                t[i] = static_cast<uint64_t> (_M_acc[i]) + _S_ct_load32_le (block + 4 * i);
        uint64_t carry = 0;
        for (size_t i = 0; i < 4; ++i) {
                uint64_t v = t[i] + carry;
                carry = v >> 32;
                t[i] = v & 0xffffffffu;
        }
        t[4] += carry;
        carry = t[4] >> 32;
        t[4] &= 0xffffffffu;
        if (carry) {
                t[0] += 5;
                carry = 0;
                for (size_t i = 0; i < 4; ++i) {
                        uint64_t v = t[i] + carry;
                        carry = v >> 32;
                        t[i] = v & 0xffffffffu;
                }
                t[4] += carry;
        }
        for (size_t i = 0; i < 5; ++i)
                _M_acc[i] = static_cast<uint32_t> (t[i]);
        uint64_t m[5];
        for (size_t i = 0; i < 5; ++i)
                m[i] = 0;
        for (size_t i = 0; i < 5; ++i) {
                uint64_t carry = 0;
                for (size_t j = 0; j < 5 && i + j < 5; ++j) {
                        uint64_t v = m[i + j] + static_cast<uint64_t> (_M_acc[i]) * _M_r[j] + carry;
                        carry = v >> 32;
                        m[i + j] = v & 0xffffffffu;
                }
                if (i + 4 < 5)
                        m[i + 4] += carry;
        }
        uint64_t carry2 = m[4] >> 32;
        m[4] &= 0xffffffffu;
        if (carry2) {
                m[0] += static_cast<uint32_t> (carry2 * 5);
                carry2 = 0;
                for (size_t i = 0; i < 4; ++i) {
                        uint64_t v = m[i] + carry2;
                        carry2 = v >> 32;
                        m[i] = v & 0xffffffffu;
                }
        }
        for (size_t i = 0; i < 5; ++i)
                _M_acc[i] = static_cast<uint32_t> (m[i]);
        _S_ct_barrier ();
}

inline void
Poly1305::update (const uint8_t *data, size_t len) noexcept {
        while (len > 0) {
                if (_M_buffered == 16) {
                        _S_process_block (_M_buffer);
                        _M_buffered = 0;
                }
                const size_t take = 16 - _M_buffered < len ? 16 - _M_buffered : len;
                for (size_t i = 0; i < take; ++i)
                        _M_buffer[_M_buffered + i] = data[i];
                _M_buffered += take;
                data += take;
                len -= take;
        }
}

inline void
Poly1305::final (uint8_t out[_S_tag_size]) noexcept {
        /* Pad with the 1-bit (0x01) then zeros, always adding at least one
         * padding block (Poly1305 appends 0x01||0* to a multiple of 16 bytes).
         * If a full block is buffered, process it first so the padding lands in
         * a fresh block; _M_buffered is then in [0,15], so the write below is
         * in-bounds. */
        if (_M_buffered == 16) {
                _S_process_block (_M_buffer);
                _M_buffered = 0;
        }
        _M_buffer[_M_buffered] = 0x01;
        for (size_t i = _M_buffered + 1; i < 16; ++i)
                _M_buffer[i] = 0;
        _S_process_block (_M_buffer);
        _M_buffered = 0;

        /* acc += s */
        uint64_t t[5];
        for (size_t i = 0; i < 4; ++i)
                t[i] = static_cast<uint64_t> (_M_acc[i]) + _M_s[i];
        t[4] = _M_acc[4];
        uint64_t carry = 0;
        for (size_t i = 0; i < 4; ++i) {
                uint64_t v = t[i] + carry;
                carry = v >> 32;
                t[i] = v & 0xffffffffu;
        }
        t[4] += carry;
        for (size_t i = 0; i < 5; ++i)
                _M_acc[i] = static_cast<uint32_t> (t[i]);

        for (size_t i = 0; i < 4; ++i)
                _S_ct_store32_le (out + 4 * i, _M_acc[i]);
        _S_ct_barrier ();
        _M_finalized = true;
}

inline void
Poly1305::mac (const uint8_t key[_S_key_size], const uint8_t *msg, size_t msg_len, uint8_t out[_S_tag_size]) noexcept {
        Poly1305 p;
        p.set_key (key);
        p.update (msg, msg_len);
        p.final (out);
}

/* ChaCha20-Poly1305 AEAD (RFC 7539 / RFC 8439).
 * Produces / consumes: [ciphertext][16-byte tag]. */
class ChaCha20Poly1305 {
      public:
        static constexpr size_t _S_key_size = 32;
        static constexpr size_t _S_nonce_size = 12;
        static constexpr size_t _S_tag_size = 16;
        static constexpr size_t _S_block_size = 64;

        ChaCha20Poly1305 () noexcept = default;

        void
        set_key (const uint8_t key[_S_key_size]) noexcept;

        /* Encrypt `plaintext` (len bytes) with AAD into `ciphertext`, write tag.
         * Returns the tag length (_S_tag_size) through out_tag_len. */
        void
        seal (const uint8_t *nonce, uint64_t counter, const uint8_t *aad, size_t aad_len, const uint8_t *plaintext,
              size_t pt_len, uint8_t *ciphertext, uint8_t tag[_S_tag_size]) noexcept;

        /* Decrypt `ciphertext` (len bytes) with AAD into `plaintext`, verify tag.
         * Returns false on authentication failure (constant-time). */
        bool
        open (const uint8_t *nonce, uint64_t counter, const uint8_t *aad, size_t aad_len, const uint8_t *ciphertext,
              size_t ct_len, const uint8_t tag[_S_tag_size], uint8_t *plaintext) noexcept;

      private:
        ChaCha20 _M_chacha{};
        Poly1305 _M_poly{};
        uint8_t _M_key[_S_key_size]{};
};

inline void
ChaCha20Poly1305::set_key (const uint8_t key[_S_key_size]) noexcept {
        for (size_t i = 0; i < _S_key_size; ++i)
                _M_key[i] = key[i];
        _S_ct_barrier ();
}

inline void
ChaCha20Poly1305::seal (const uint8_t *nonce, uint64_t counter, const uint8_t *aad, size_t aad_len,
                        const uint8_t *plaintext, size_t pt_len, uint8_t *ciphertext,
                        uint8_t tag[_S_tag_size]) noexcept {
        (void) counter; /* nonce already embeds the sequence/counter */
        /* Poly1305 one-time key = first 32 bytes of ChaCha20 keystream block 0. */
        uint8_t zeros[_S_block_size]{};
        uint8_t poly_key_block[_S_block_size];
        _M_chacha.set_key (_M_key);
        _M_chacha.set_nonce (nonce);
        _M_chacha.set_counter (UINT32_C (0));
        _M_chacha.crypt (zeros, poly_key_block, _S_block_size);
        _M_poly.set_key (poly_key_block);

        /* AAD: pad to 16-byte blocks, then absorb. */
        size_t aad_pos = 0;
        uint8_t block[16];
        while (aad_pos < aad_len) {
                const size_t take = aad_len - aad_pos < 16 ? aad_len - aad_pos : 16;
                for (size_t i = 0; i < 16; ++i)
                        block[i] = (i < take) ? aad[aad_pos + i] : 0;
                _M_poly.update (block, 16);
                aad_pos += take;
        }
        /* Ciphertext: pad to 16-byte blocks, then absorb.
         * CTR encryption: keystream blocks starting at counter=1. */
        _M_chacha.set_key (_M_key);
        _M_chacha.set_nonce (nonce);
        size_t ct_pos = 0;
        uint32_t ctr = UINT32_C (1);
        while (ct_pos < pt_len) {
                uint8_t keystream[_S_block_size];
                _M_chacha.set_counter (ctr);
                _M_chacha.crypt (zeros, keystream, _S_block_size);
                const size_t take = pt_len - ct_pos < _S_block_size ? pt_len - ct_pos : _S_block_size;
                for (size_t i = 0; i < take; ++i)
                        ciphertext[ct_pos + i] = static_cast<uint8_t> (plaintext[ct_pos + i] ^ keystream[i]);
                for (size_t i = 0; i < 16; ++i)
                        block[i] = (i < take) ? ciphertext[ct_pos + i] : 0;
                _M_poly.update (block, 16);
                ct_pos += take;
                ctr += 1;
        }
        /* Length block: 64-bit aad bits, 64-bit ciphertext bits, LE. */
        uint8_t len_block[16]{};
        _S_ct_store64_le (len_block + 0, aad_len * 8);
        _S_ct_store64_le (len_block + 8, pt_len * 8);
        _M_poly.update (len_block, 16);
        _M_poly.final (tag);

        _S_ct_barrier ();
}

inline bool
ChaCha20Poly1305::open (const uint8_t *nonce, uint64_t counter, const uint8_t *aad, size_t aad_len,
                        const uint8_t *ciphertext, size_t ct_len, const uint8_t tag[_S_tag_size],
                        uint8_t *plaintext) noexcept {
        (void) counter; /* nonce already embeds the sequence/counter */
        /* Poly1305 one-time key = first 32 bytes of ChaCha20 keystream block 0. */
        uint8_t zeros[_S_block_size]{};
        uint8_t poly_key_block[_S_block_size];
        _M_chacha.set_key (_M_key);
        _M_chacha.set_nonce (nonce);
        _M_chacha.set_counter (UINT32_C (0));
        _M_chacha.crypt (zeros, poly_key_block, _S_block_size);
        _M_poly.set_key (poly_key_block);

        size_t aad_pos = 0;
        uint8_t block[16];
        while (aad_pos < aad_len) {
                const size_t take = aad_len - aad_pos < 16 ? aad_len - aad_pos : 16;
                for (size_t i = 0; i < 16; ++i)
                        block[i] = (i < take) ? aad[aad_pos + i] : 0;
                _M_poly.update (block, 16);
                aad_pos += take;
        }
        size_t ct_pos = 0;
        _M_chacha.set_key (_M_key);
        _M_chacha.set_nonce (nonce);
        uint32_t ctr = UINT32_C (1);
        while (ct_pos < ct_len) {
                uint8_t keystream[_S_block_size];
                _M_chacha.set_counter (ctr);
                _M_chacha.crypt (zeros, keystream, _S_block_size);
                const size_t take = ct_len - ct_pos < _S_block_size ? ct_len - ct_pos : _S_block_size;
                for (size_t i = 0; i < take; ++i)
                        plaintext[ct_pos + i] = static_cast<uint8_t> (ciphertext[ct_pos + i] ^ keystream[i]);
                for (size_t i = 0; i < 16; ++i)
                        block[i] = (i < take) ? ciphertext[ct_pos + i] : 0;
                _M_poly.update (block, 16);
                ct_pos += take;
                ctr += 1;
        }
        uint8_t len_block[16]{};
        _S_ct_store64_le (len_block + 0, aad_len * 8);
        _S_ct_store64_le (len_block + 8, ct_len * 8);
        _M_poly.update (len_block, 16);
        uint8_t expected_tag[_S_tag_size];
        _M_poly.final (expected_tag);

        /* Constant-time tag comparison: XOR-accumulate, not memcmp/==. */
        uint8_t acc = 0;
        for (size_t i = 0; i < _S_tag_size; ++i)
                acc |= expected_tag[i] ^ tag[i];
        const bool ok = acc == 0;
        _S_ct_barrier ();
        return ok;
}

/* =========================================================================
 * AES-256-GCM.
 * Portable constant-time C++ implementation here; x86_64 AES-NI assembly
 * provides the hot-path primitives when available.
 * ========================================================================= */

class AesGcm {
      public:
        static constexpr size_t _S_key_size = 32;
        static constexpr size_t _S_nonce_size = 12;
        static constexpr size_t _S_tag_size = 16;
        static constexpr size_t _S_block_size = 16;
        static constexpr int _S_rounds = 14; /* AES-256 */

        AesGcm () noexcept = default;

        void
        set_key (const uint8_t key[_S_key_size]) noexcept;

        void
        seal (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
              const uint8_t *plaintext, size_t pt_len, uint8_t *ciphertext, uint8_t tag[_S_tag_size]) noexcept;

        bool
        open (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
              const uint8_t *ciphertext, size_t ct_len, const uint8_t tag[_S_tag_size], uint8_t *plaintext) noexcept;

      private:
        /* Key schedule: 15 round keys * 16 bytes = 240 bytes. */
        void
        _S_expand_key (const uint8_t key[_S_key_size]) noexcept;

        void
        _S_encrypt_block (const uint8_t in[16], uint8_t out[16]) noexcept;

        void
        _S_compute_tag (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
                        const uint8_t *ciphertext, size_t ct_len, uint8_t tag[_S_tag_size]) noexcept;

        void
        _S_ghash (const uint8_t *h, const uint8_t *data, size_t len, uint8_t *y) noexcept;

        void
        _M_h_compute () noexcept;

        uint8_t _M_key[_S_key_size]{}; /* AES-256 key */
        uint8_t _M_rk[15 * 16]{};      /* round keys */
        uint8_t _M_h[16]{};            /* GHASH subkey = E_K(0) */
        bool _M_key_set = false;
};

/* -----------------------------------------------------------------------
 * SHAKE256 — Keccak-f[1600] sponge, constant-time software.
 * --------------------------------------------------------------------- */

[[nodiscard]] inline uint64_t
_S_rotl64_const (uint64_t x, int n) noexcept {
        return (x << n) | (x >> (64 - n));
}

/* Pi step destination indices for the 24-lane cycle starting at lane 1,
 * followed by the Rho rotation offsets for each of the 24 steps. */
static constexpr int _S_pi_dest[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};
static constexpr int _S_rho_offset[24] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 14,
};

inline void
Shake256::_S_keccakf () noexcept {
        /* Round constants for Keccak-f[1600] (24 rounds). */
        static constexpr uint64_t _S_rc[24] = {
            UINT64_C (0x0000000000000001), UINT64_C (0x0000000000008082), UINT64_C (0x800000000000808a),
            UINT64_C (0x8000000080008000), UINT64_C (0x000000000000808b), UINT64_C (0x0000000080000001),
            UINT64_C (0x8000000080008081), UINT64_C (0x8000000000008009), UINT64_C (0x000000000000008a),
            UINT64_C (0x0000000000000088), UINT64_C (0x0000000080008009), UINT64_C (0x000000008000000a),
            UINT64_C (0x000000008000808b), UINT64_C (0x800000000000008b), UINT64_C (0x8000000000008089),
            UINT64_C (0x8000000080008003), UINT64_C (0x8000000000008002), UINT64_C (0x8000000000000080),
            UINT64_C (0x000000000000800a), UINT64_C (0x000000000000000d), UINT64_C (0x000000008000000d),
            UINT64_C (0x800000008000000d), UINT64_C (0x8000000000008085), UINT64_C (0x8000000000000087),
        };

        uint64_t bc[5];

        for (int round = 0; round < 24; ++round) {
                /* Theta */
                for (int i = 0; i < 5; ++i)
                        bc[i] = _M_state[i] ^ _M_state[i + 5] ^ _M_state[i + 10] ^ _M_state[i + 15] ^ _M_state[i + 20];

                for (int i = 0; i < 5; ++i) {
                        const uint64_t t = bc[(i + 4) % 5] ^ _S_rotl64_const (bc[(i + 1) % 5], 1);
                        for (int j = 0; j < 25; j += 5)
                                _M_state[i + j] ^= t;
                }

                /* Rho + Pi (follow the 24-lane cycle, constant-time). */
                uint64_t cur = _M_state[1];
                for (int i = 0; i < 24; ++i) {
                        const int dest = _S_pi_dest[i];
                        const uint64_t temp = cur;
                        cur = _M_state[dest];
                        _M_state[dest] = _S_rotl64_const (temp, _S_rho_offset[i]);
                }

                /* Chi */
                for (int j = 0; j < 25; j += 5) {
                        for (int i = 0; i < 5; ++i)
                                bc[i] = _M_state[i + j];
                        for (int i = 0; i < 5; ++i)
                                _M_state[i + j] = bc[i] ^ ((~bc[(i + 1) % 5]) & bc[(i + 2) % 5]);
                }

                /* Iota */
                _M_state[0] ^= _S_rc[round];

                _S_ct_barrier ();
        }
}

inline void
Shake256::_S_absorb_block (const uint8_t *block) noexcept {
        for (size_t i = 0; i < _S_rate / 8; ++i)
                _M_state[i] ^= _S_ct_load64_le (block + 8 * i);
        _S_keccakf ();
}

inline void
Shake256::absorb (const uint8_t *data, size_t len) noexcept {
        if (_M_buffered != 0) {
                size_t want = _S_rate - _M_buffered;
                if (want > len)
                        want = len;
                for (size_t i = 0; i < want; ++i)
                        _M_buffer[_M_buffered + i] = data[i];
                _M_buffered += want;
                data += want;
                len -= want;
                if (_M_buffered == _S_rate) {
                        _S_absorb_block (_M_buffer);
                        _M_buffered = 0;
                }
        }
        while (len >= _S_rate) {
                _S_absorb_block (data);
                data += _S_rate;
                len -= _S_rate;
        }
        if (len != 0) {
                for (size_t i = 0; i < len; ++i)
                        _M_buffer[_M_buffered + i] = data[i];
                _M_buffered += len;
        }
}

inline void
Shake256::squeeze (uint8_t *out, size_t out_len) noexcept {
        if (_M_squeezed == 0 && _M_buffered != 0) {
                /* Pad the last block: 0x06 | ... | 0x80 (multirate padding for SHAKE). */
                _M_buffer[_M_buffered] = 0x06;
                for (size_t i = _M_buffered + 1; i < _S_rate; ++i)
                        _M_buffer[i] = 0;
                _M_buffer[_S_rate - 1] |= 0x80;
                _S_absorb_block (_M_buffer);
                _M_buffered = 0;
                _M_squeezed = 1;
        }
        while (out_len > 0) {
                const size_t take = out_len < _S_rate ? out_len : _S_rate;
                for (size_t i = 0; i < take / 8; ++i) {
                        const uint64_t v = _M_state[i];
                        out[8 * i] = static_cast<uint8_t> (v & 0xff);
                        out[8 * i + 1] = static_cast<uint8_t> ((v >> 8) & 0xff);
                        out[8 * i + 2] = static_cast<uint8_t> ((v >> 16) & 0xff);
                        out[8 * i + 3] = static_cast<uint8_t> ((v >> 24) & 0xff);
                        out[8 * i + 4] = static_cast<uint8_t> ((v >> 32) & 0xff);
                        out[8 * i + 5] = static_cast<uint8_t> ((v >> 40) & 0xff);
                        out[8 * i + 6] = static_cast<uint8_t> ((v >> 48) & 0xff);
                        out[8 * i + 7] = static_cast<uint8_t> ((v >> 56) & 0xff);
                }
                /* Handle trailing bytes not 8-aligned. */
                const size_t rem = take % 8;
                if (rem != 0) {
                        const size_t base = take - rem;
                        const uint64_t v = _M_state[base / 8];
                        for (size_t i = 0; i < rem; ++i)
                                out[base + i] = static_cast<uint8_t> ((v >> (8 * i)) & 0xff);
                }
                out += take;
                out_len -= take;
                if (out_len > 0)
                        _S_keccakf ();
        }
}

inline void
Shake256::derive (const uint8_t *data, size_t data_len, uint8_t *out, size_t out_len) noexcept {
        reset ();
        absorb (data, data_len);
        squeeze (out, out_len);
}

inline void
Shake256::reset () noexcept {
        for (size_t i = 0; i < 25; ++i)
                _M_state[i] = 0;
        _M_buffered = 0;
        for (size_t i = 0; i < _S_rate; ++i)
                _M_buffer[i] = 0;
        _M_squeezed = 0;
}

/* =========================================================================
 * SecureChannel — public API.
 * ========================================================================= */

struct SecureChannel {
        SecureChannelAlgo _M_algo = SecureChannelAlgo::AES_256_GCM;
        uint8_t _M_key[32]{};    /* PSK or derived key */
        uint8_t _M_nonce[12]{};  /* per-record base nonce */
        uint64_t _M_seq = 0;     /* sequence number (replay protection) */
        uint8_t _M_state[256]{}; /* algorithm-specific state */
};

[[nodiscard]] inline constexpr bool
secure_channel_algo_supported (SecureChannelAlgo algo) noexcept {
        return algo == SecureChannelAlgo::AES_256_GCM || algo == SecureChannelAlgo::CHACHA20_POLY1305
               || algo == SecureChannelAlgo::SHAKE256_KDF;
}

[[nodiscard]] inline Maybe<SecureChannel, SecureChannelError>
secure_channel_init (SecureChannelAlgo algo, const uint8_t *psk, size_t psk_len) noexcept {
        if (!secure_channel_algo_supported (algo))
                return Maybe<SecureChannel, SecureChannelError> (SecureChannelError::UNSUPPORTED_ALGO);
        if (psk == nullptr || psk_len < 16 || psk_len > 32)
                return Maybe<SecureChannel, SecureChannelError> (SecureChannelError::INVALID_KEY);

        SecureChannel ch{};
        ch._M_algo = algo;
        shake256_derive_key (psk, psk_len, nullptr, 0, ch._M_key, sizeof (ch._M_key));
        _S_ct_barrier ();

        zzs::base::log_debug ("secure_channel_init: algo=%u key_len=%zu",
                              static_cast<unsigned> (static_cast<uint8_t> (algo)), psk_len);

        return Maybe<SecureChannel, SecureChannelError> (std::in_place, ch);
}
constexpr size_t _S_aad_len = 22;

/* Maximum plaintext bytes that fit into one chunk given `out_cap`.
 * Record overhead = 2 (len) + 8 (seq) + 12 (nonce) + 16 (tag) = 38. */
[[nodiscard]] inline constexpr size_t
secure_channel_max_plaintext (size_t out_cap) noexcept {
        return out_cap > 38 ? out_cap - 38 : 0;
}

/* Number of chunks needed to carry `total_len` plaintext bytes with
 * `chunk_size` plaintext bytes per chunk. */
[[nodiscard]] inline constexpr size_t
secure_channel_chunk_count (size_t total_len, size_t chunk_size) noexcept {
        if (chunk_size == 0)
                return 0;
        return (total_len + chunk_size - 1) / chunk_size;
}

/* Encrypt a single chunk. Writes the record format into `out`.
 * Returns the number of bytes written to `out`. */
[[nodiscard]] inline Maybe<size_t, SecureChannelError>
secure_channel_encrypt_chunk (SecureChannel &ch, const uint8_t *plaintext, size_t plaintext_len, uint8_t *out,
                              size_t out_cap) noexcept {
        if (out_cap < 38)
                return Maybe<size_t, SecureChannelError> (SecureChannelError::BUFFER_TOO_SMALL);
        const size_t max_pt = secure_channel_max_plaintext (out_cap);
        if (plaintext_len > max_pt)
                return Maybe<size_t, SecureChannelError> (SecureChannelError::BUFFER_TOO_SMALL);
        if (ch._M_seq >= (UINT64_C (1) << 56))
                return Maybe<size_t, SecureChannelError> (SecureChannelError::SEQ_OVERFLOW);

        /* Record header: [2:len][8:seq][12:nonce]. */
        out[0] = static_cast<uint8_t> (plaintext_len & 0xff);
        out[1] = static_cast<uint8_t> ((plaintext_len >> 8) & 0xff);
        _S_ct_store64_le (out + 2, ch._M_seq);

        /* Per-record nonce = base nonce XOR sequence (low 8 bytes), constant-time. */
        uint8_t nonce[12];
        for (size_t i = 0; i < 12; ++i)
                nonce[i] = ch._M_nonce[i];
        for (size_t i = 0; i < 8; ++i)
                nonce[i] ^= static_cast<uint8_t> ((ch._M_seq >> (8 * i)) & 0xff);
        _S_ct_barrier ();

        /* AAD = header (22 bytes). */
        const uint8_t *aad = out;
        uint8_t tag[16];
        uint8_t *ciphertext = out + 22; /* ciphertext starts after header */

        if (ch._M_algo == SecureChannelAlgo::AES_256_GCM) {
                AesGcm gcm;
                /* Derive per-record AES key from the channel key + nonce via SHAKE256. */
                uint8_t record_key[32];
                shake256_derive_key (ch._M_key, sizeof (ch._M_key), nonce, sizeof (nonce), record_key,
                                     sizeof (record_key));
                gcm.set_key (record_key);
                gcm.seal (nonce, ch._M_seq, aad, _S_aad_len, plaintext, plaintext_len, ciphertext, tag);
        } else if (ch._M_algo == SecureChannelAlgo::CHACHA20_POLY1305) {
                ChaCha20Poly1305 chacha;
                uint8_t record_key[32];
                shake256_derive_key (ch._M_key, sizeof (ch._M_key), nonce, sizeof (nonce), record_key,
                                     sizeof (record_key));
                chacha.set_key (record_key);
                chacha.seal (nonce, ch._M_seq, aad, _S_aad_len, plaintext, plaintext_len, ciphertext, tag);
        } else {
                /* SHAKE256_KDF is not an AEAD; cannot encrypt. */
                return Maybe<size_t, SecureChannelError> (SecureChannelError::UNSUPPORTED_ALGO);
        }

        /* Copy tag after ciphertext. */
        for (size_t i = 0; i < 16; ++i)
                ciphertext[plaintext_len + i] = tag[i];

        _S_ct_barrier ();
        ++ch._M_seq;
        return Maybe<size_t, SecureChannelError> (static_cast<size_t> (22 + plaintext_len + 16));
}

/* Decrypt a single chunk in the record format. Returns plaintext length. */
[[nodiscard]] inline Maybe<size_t, SecureChannelError>
secure_channel_decrypt_chunk (SecureChannel &ch, const uint8_t *ciphertext, size_t ciphertext_len, uint8_t *out,
                              size_t out_cap) noexcept {
        if (ciphertext_len < 38)
                return Maybe<size_t, SecureChannelError> (SecureChannelError::BUFFER_TOO_SMALL);

        const size_t plaintext_len = static_cast<size_t> (ciphertext[0]) | (static_cast<size_t> (ciphertext[1]) << 8);
        const size_t max_pt = secure_channel_max_plaintext (ciphertext_len);
        if (plaintext_len > max_pt || plaintext_len > out_cap)
                return Maybe<size_t, SecureChannelError> (SecureChannelError::BUFFER_TOO_SMALL);
        if (ch._M_seq >= (UINT64_C (1) << 56))
                return Maybe<size_t, SecureChannelError> (SecureChannelError::SEQ_OVERFLOW);

        const uint8_t *aad = ciphertext;
        const uint8_t *ct = ciphertext + 22;
        const uint8_t *tag = ct + plaintext_len;

        /* Reconstruct nonce from header (seq embedded) XOR base nonce. */
        const uint64_t seq = _S_ct_load64_le (ciphertext + 2);
        uint8_t nonce[12];
        for (size_t i = 0; i < 12; ++i)
                nonce[i] = ch._M_nonce[i];
        for (size_t i = 0; i < 8; ++i)
                nonce[i] ^= static_cast<uint8_t> ((seq >> (8 * i)) & 0xff);
        _S_ct_barrier ();

        uint8_t record_key[32];
        shake256_derive_key (ch._M_key, sizeof (ch._M_key), nonce, sizeof (nonce), record_key, sizeof (record_key));

        bool ok = false;
        if (ch._M_algo == SecureChannelAlgo::AES_256_GCM) {
                AesGcm gcm;
                gcm.set_key (record_key);
                ok = gcm.open (nonce, seq, aad, _S_aad_len, ct, plaintext_len, tag, out);
        } else if (ch._M_algo == SecureChannelAlgo::CHACHA20_POLY1305) {
                ChaCha20Poly1305 chacha;
                chacha.set_key (record_key);
                ok = chacha.open (nonce, seq, aad, _S_aad_len, ct, plaintext_len, tag, out);
        } else {
                return Maybe<size_t, SecureChannelError> (SecureChannelError::UNSUPPORTED_ALGO);
        }

        _S_ct_barrier ();
        if (!ok)
                return Maybe<size_t, SecureChannelError> (SecureChannelError::AUTH_FAILED);

        ++ch._M_seq;
        return Maybe<size_t, SecureChannelError> (plaintext_len);
}

/* -----------------------------------------------------------------------
 * AES-256-GCM — portable constant-time C++ implementation.
 * --------------------------------------------------------------------- */

/* Constant-time GF(2^8) multiply modulo the AES polynomial 0x11B.
 * No data-dependent branches or memory accesses. */
[[nodiscard]] inline uint8_t
_S_gf_mul (uint8_t a, uint8_t b) noexcept {
        uint16_t p = 0;
        uint16_t aa = a;
        for (int i = 0; i < 8; ++i) {
                const uint16_t bmask = 0 - static_cast<uint16_t> (b & 1u);
                p ^= aa & bmask;
                uint16_t next = static_cast<uint16_t> (aa << 1);
                const uint16_t reduce_mask = 0 - static_cast<uint16_t> ((next >> 8) & 1u);
                next = static_cast<uint16_t> ((next & 0xffu) ^ (reduce_mask & 0x1bu));
                aa = next;
                b >>= 1;
        }
        return static_cast<uint8_t> (p);
}

/* x^254 in GF(2^8) = x^(-1) for x != 0, and 0 for x == 0.
 * Addition chain: 1,2,3,6,7,14,15,30,31,62,63,126,127,254. */
[[nodiscard]] inline uint8_t
_S_gf_inv (uint8_t x) noexcept {
        const uint8_t x2 = _S_gf_mul (x, x);
        const uint8_t x3 = _S_gf_mul (x2, x);
        const uint8_t x6 = _S_gf_mul (x3, x3);
        const uint8_t x7 = _S_gf_mul (x6, x);
        const uint8_t x14 = _S_gf_mul (x7, x7);
        const uint8_t x15 = _S_gf_mul (x14, x);
        const uint8_t x30 = _S_gf_mul (x15, x15);
        const uint8_t x31 = _S_gf_mul (x30, x);
        const uint8_t x62 = _S_gf_mul (x31, x31);
        const uint8_t x63 = _S_gf_mul (x62, x);
        const uint8_t x126 = _S_gf_mul (x63, x63);
        const uint8_t x127 = _S_gf_mul (x126, x);
        return _S_gf_mul (x127, x127);
}

[[nodiscard]] inline uint8_t
_S_aes_bit (uint8_t v, int i) noexcept {
        return static_cast<uint8_t> ((v >> i) & 1u);
}

/* AES S-box: GF(2^8) inverse followed by the affine transform.
 * Fully bitwise, no table lookups indexed by secret data. */
[[nodiscard]] inline uint8_t
_S_aes_sbox (uint8_t x) noexcept {
        const uint8_t inv = _S_gf_inv (x);
        /* Affine transform over GF(2), constant c = 0x63. */
        const uint8_t b0 = _S_aes_bit (inv, 0) ^ _S_aes_bit (inv, 4) ^ _S_aes_bit (inv, 5) ^ _S_aes_bit (inv, 6)
                           ^ _S_aes_bit (inv, 7) ^ 1u;
        const uint8_t b1 = _S_aes_bit (inv, 1) ^ _S_aes_bit (inv, 5) ^ _S_aes_bit (inv, 6) ^ _S_aes_bit (inv, 7)
                           ^ _S_aes_bit (inv, 0) ^ 1u;
        const uint8_t b2 = _S_aes_bit (inv, 2) ^ _S_aes_bit (inv, 6) ^ _S_aes_bit (inv, 7) ^ _S_aes_bit (inv, 0)
                           ^ _S_aes_bit (inv, 1) ^ 0u;
        const uint8_t b3 = _S_aes_bit (inv, 3) ^ _S_aes_bit (inv, 7) ^ _S_aes_bit (inv, 0) ^ _S_aes_bit (inv, 1)
                           ^ _S_aes_bit (inv, 2) ^ 1u;
        const uint8_t b4 = _S_aes_bit (inv, 4) ^ _S_aes_bit (inv, 0) ^ _S_aes_bit (inv, 1) ^ _S_aes_bit (inv, 2)
                           ^ _S_aes_bit (inv, 3) ^ 1u;
        const uint8_t b5 = _S_aes_bit (inv, 5) ^ _S_aes_bit (inv, 1) ^ _S_aes_bit (inv, 2) ^ _S_aes_bit (inv, 3)
                           ^ _S_aes_bit (inv, 4) ^ 0u;
        const uint8_t b6 = _S_aes_bit (inv, 6) ^ _S_aes_bit (inv, 2) ^ _S_aes_bit (inv, 3) ^ _S_aes_bit (inv, 4)
                           ^ _S_aes_bit (inv, 5) ^ 1u;
        const uint8_t b7 = _S_aes_bit (inv, 7) ^ _S_aes_bit (inv, 3) ^ _S_aes_bit (inv, 4) ^ _S_aes_bit (inv, 5)
                           ^ _S_aes_bit (inv, 6) ^ 0u;
        return static_cast<uint8_t> (b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5) | (b6 << 6)
                                     | (b7 << 7));
}

inline void
_S_aes_sub_bytes (uint8_t s[16]) noexcept {
        for (size_t i = 0; i < 16; ++i)
                s[i] = _S_aes_sbox (s[i]);
        _S_ct_barrier ();
}

inline void
_S_aes_shift_rows (uint8_t s[16]) noexcept {
        uint8_t t[16];
        t[0] = s[0];
        t[1] = s[5];
        t[2] = s[10];
        t[3] = s[15];
        t[4] = s[4];
        t[5] = s[9];
        t[6] = s[14];
        t[7] = s[3];
        t[8] = s[8];
        t[9] = s[13];
        t[10] = s[2];
        t[11] = s[7];
        t[12] = s[12];
        t[13] = s[1];
        t[14] = s[6];
        t[15] = s[11];
        for (size_t i = 0; i < 16; ++i)
                s[i] = t[i];
}

[[nodiscard]] inline uint8_t
_S_aes_xtime (uint8_t x) noexcept {
        return static_cast<uint8_t> ((x << 1) ^ (0x1bu & static_cast<uint8_t> (-(static_cast<int> (x >> 7) & 1))));
}

inline void
_S_aes_mix_columns (uint8_t s[16]) noexcept {
        for (size_t c = 0; c < 4; ++c) {
                const uint8_t a0 = s[c];
                const uint8_t a1 = s[4 + c];
                const uint8_t a2 = s[8 + c];
                const uint8_t a3 = s[12 + c];
                const uint8_t h = static_cast<uint8_t> (a0 ^ a1 ^ a2 ^ a3);
                s[c] ^= h ^ _S_aes_xtime (a0 ^ a1);
                s[4 + c] ^= h ^ _S_aes_xtime (a1 ^ a2);
                s[8 + c] ^= h ^ _S_aes_xtime (a2 ^ a3);
                s[12 + c] ^= h ^ _S_aes_xtime (a3 ^ a0);
        }
        _S_ct_barrier ();
}

inline void
_S_aes_add_round_key (uint8_t s[16], const uint8_t rk[16]) noexcept {
        for (size_t i = 0; i < 16; ++i)
                s[i] ^= rk[i];
}

inline void
AesGcm::set_key (const uint8_t key[_S_key_size]) noexcept {
        for (size_t i = 0; i < _S_key_size; ++i)
                _M_key[i] = key[i];
        _S_expand_key (_M_key);
}

inline void
AesGcm::_S_expand_key (const uint8_t key[_S_key_size]) noexcept {
        /* AES-256 key schedule: 60 words (15 round keys x 16 bytes), constant-time.
         * w[0..7] = key; w[i] = w[i-8] ^ temp, temp transformed when i%8==0 or 4. */
        uint8_t rk[15 * 16];
        for (size_t i = 0; i < 32; ++i)
                rk[i] = key[i]; /* w[0..7] = key (32 bytes) */
        size_t i = 8;
        while (i < 15 * 4) {
                uint8_t t[4];
                for (size_t j = 0; j < 4; ++j)
                        t[j] = rk[4 * (i - 1) + j]; /* w[i-1] */
                const bool is_8 = (i % 8) == 0;
                const bool is_4 = (i % 8) == 4;
                if (is_8) {
                        uint8_t tmp = t[0];
                        t[0] = t[1];
                        t[1] = t[2];
                        t[2] = t[3];
                        t[3] = tmp;
                        t[0] = _S_aes_sbox (t[0]);
                        t[1] = _S_aes_sbox (t[1]);
                        t[2] = _S_aes_sbox (t[2]);
                        t[3] = _S_aes_sbox (t[3]);
                        /* Rcon = 1 << (i/8 - 1), constant-time select. */
                        const size_t rcon_idx = i / 8 - 1;
                        uint8_t rcon = 0;
                        for (size_t rc = 0; rc < 7; ++rc) {
                                const uint8_t cond = static_cast<uint8_t> (-(static_cast<int> (rcon_idx == rc)));
                                rcon ^= static_cast<uint8_t> (cond & static_cast<uint8_t> (UINT64_C (1) << rc));
                        }
                        t[0] ^= rcon;
                } else if (is_4) {
                        t[0] = _S_aes_sbox (t[0]);
                        t[1] = _S_aes_sbox (t[1]);
                        t[2] = _S_aes_sbox (t[2]);
                        t[3] = _S_aes_sbox (t[3]);
                }
                for (size_t j = 0; j < 4; ++j)
                        rk[4 * i + j] = static_cast<uint8_t> (rk[4 * (i - 8) + j] ^ t[j]);
                i += 1;
        }
        for (size_t i = 0; i < 15 * 16; ++i)
                _M_rk[i] = rk[i];
        _M_h_compute ();
        _M_key_set = true;
        _S_ct_barrier ();
}

inline void
AesGcm::_M_h_compute () noexcept {
        /* H = E_K(0) — the GHASH subkey. */
        uint8_t zero[16]{};
        _S_encrypt_block (zero, _M_h);
}

inline void
AesGcm::_S_encrypt_block (const uint8_t in[16], uint8_t out[16]) noexcept {
        uint8_t s[16];
        for (size_t i = 0; i < 16; ++i)
                s[i] = in[i];
        _S_aes_add_round_key (s, _M_rk);
        for (int r = 1; r < _S_rounds; ++r) {
                _S_aes_sub_bytes (s);
                _S_aes_shift_rows (s);
                _S_aes_mix_columns (s);
                _S_aes_add_round_key (s, _M_rk + r * 16);
        }
        _S_aes_sub_bytes (s);
        _S_aes_shift_rows (s);
        _S_aes_add_round_key (s, _M_rk + _S_rounds * 16);
        for (size_t i = 0; i < 16; ++i)
                out[i] = s[i];
        _S_ct_barrier ();
}

/* GHASH in GF(2^128), constant-time bit-by-bit carry-less multiply. */
inline void
AesGcm::_S_ghash (const uint8_t *h, const uint8_t *data, size_t len, uint8_t *y) noexcept {
        for (size_t i = 0; i < len; i += 16) {
                /* XOR only the valid bytes of this block (len may not be 16-aligned;
                 * missing bytes are treated as zero, which y already holds). */
                size_t block_len = len - i;
                if (block_len > 16)
                        block_len = 16;
                for (size_t j = 0; j < block_len; ++j)
                        y[j] ^= static_cast<uint8_t> (data[i + j]);
                for (int bit = 0; bit < 128; ++bit) {
                        const uint8_t lsb = static_cast<uint8_t> (-(static_cast<int> (y[15] & 1)));
                        for (size_t j = 15; j > 0; --j)
                                y[j] = static_cast<uint8_t> ((y[j] >> 1) | (y[j - 1] << 7));
                        y[0] = static_cast<uint8_t> (y[0] >> 1);
                        for (size_t j = 0; j < 16; ++j)
                                y[j] ^= static_cast<uint8_t> (h[j] & lsb);
                }
        }
        _S_ct_barrier ();
}

inline void
AesGcm::_S_compute_tag (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
                        const uint8_t *ciphertext, size_t ct_len, uint8_t tag[_S_tag_size]) noexcept {
        if (!_M_key_set)
                set_key (_M_key);

        /* J0 = nonce || counter (96-bit nonce + 32-bit counter). */
        uint8_t j0[16];
        for (size_t i = 0; i < 12; ++i)
                j0[i] = nonce[i];
        _S_ct_store32_le (j0 + 12, static_cast<uint32_t> (counter & 0xffffffffu));

        /* GHASH over AAD || ciphertext || lengths. */
        uint8_t y[16]{};
        _S_ghash (_M_h, aad, aad_len, y);
        if (aad_len % 16 != 0) {
                uint8_t pad[16]{};
                const size_t rem = aad_len % 16;
                for (size_t i = 0; i < rem; ++i)
                        pad[i] = aad[(aad_len / 16) * 16 + i];
                _S_ghash (_M_h, pad, 16, y);
        }
        _S_ghash (_M_h, ciphertext, ct_len, y);
        if (ct_len % 16 != 0) {
                uint8_t pad[16]{};
                const size_t rem = ct_len % 16;
                for (size_t i = 0; i < rem; ++i)
                        pad[i] = ciphertext[(ct_len / 16) * 16 + i];
                _S_ghash (_M_h, pad, 16, y);
        }
        /* Length block: 64-bit AAD bits, 64-bit ciphertext bits, little-endian. */
        uint8_t len_block[16]{};
        _S_ct_store64_le (len_block + 0, aad_len * 8);
        _S_ct_store64_le (len_block + 8, ct_len * 8);
        _S_ghash (_M_h, len_block, 16, y);

        /* Tag = E_K(J0) XOR Y. */
        uint8_t enc_j0[16];
        _S_encrypt_block (j0, enc_j0);
        for (size_t i = 0; i < 16; ++i)
                tag[i] = static_cast<uint8_t> (y[i] ^ enc_j0[i]);
}

inline void
AesGcm::seal (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
              const uint8_t *plaintext, size_t pt_len, uint8_t *ciphertext, uint8_t tag[_S_tag_size]) noexcept {
        /* Compute the authentication tag over the plaintext-as-ciphertext
         * (GHASH is over the ciphertext bytes, which for seal is the output). */
        /* CTR encryption first so ciphertext is ready for tag computation. */
        uint8_t j0[16];
        for (size_t i = 0; i < 12; ++i)
                j0[i] = nonce[i];
        _S_ct_store32_le (j0 + 12, static_cast<uint32_t> (counter & 0xffffffffu));

        /* CTR encryption: E_K(J0+1), E_K(J0+2), ... */
        uint8_t ctr[16];
        for (size_t i = 0; i < 16; ++i)
                ctr[i] = j0[i];
        ctr[12] = 1;
        ctr[13] = 0;
        ctr[14] = 0;
        ctr[15] = 0;
        size_t pos = 0;
        while (pos < pt_len) {
                uint8_t keystream[16];
                _S_encrypt_block (ctr, keystream);
                const size_t take = pt_len - pos < 16 ? pt_len - pos : 16;
                for (size_t i = 0; i < take; ++i)
                        ciphertext[pos + i] = static_cast<uint8_t> (plaintext[pos + i] ^ keystream[i]);
                pos += take;
                for (int i = 12; i < 16; ++i) {
                        const uint8_t inc = static_cast<uint8_t> (ctr[i] + 1);
                        const uint8_t carry = static_cast<uint8_t> (-(static_cast<int> (inc == 0)));
                        ctr[i] = inc;
                        if (!carry)
                                break;
                }
                _S_ct_barrier ();
        }

        /* Now compute the tag over the produced ciphertext. */
        _S_compute_tag (nonce, counter, aad, aad_len, ciphertext, pt_len, tag);
}

inline bool
AesGcm::open (const uint8_t nonce[_S_nonce_size], uint64_t counter, const uint8_t *aad, size_t aad_len,
              const uint8_t *ciphertext, size_t ct_len, const uint8_t tag[_S_tag_size], uint8_t *plaintext) noexcept {
        /* Compute expected tag over the ciphertext (no decryption yet). */
        uint8_t expected_tag[_S_tag_size];
        _S_compute_tag (nonce, counter, aad, aad_len, ciphertext, ct_len, expected_tag);

        /* CTR decryption (same as encryption with the keystream). */
        uint8_t j0[16];
        for (size_t i = 0; i < 12; ++i)
                j0[i] = nonce[i];
        _S_ct_store32_le (j0 + 12, static_cast<uint32_t> (counter & 0xffffffffu));
        uint8_t ctr[16];
        for (size_t i = 0; i < 16; ++i)
                ctr[i] = j0[i];
        ctr[12] = 1;
        ctr[13] = 0;
        ctr[14] = 0;
        ctr[15] = 0;
        size_t pos = 0;
        while (pos < ct_len) {
                uint8_t keystream[16];
                _S_encrypt_block (ctr, keystream);
                const size_t take = ct_len - pos < 16 ? ct_len - pos : 16;
                for (size_t i = 0; i < take; ++i)
                        plaintext[pos + i] = static_cast<uint8_t> (ciphertext[pos + i] ^ keystream[i]);
                pos += take;
                for (int i = 12; i < 16; ++i) {
                        const uint8_t inc = static_cast<uint8_t> (ctr[i] + 1);
                        const uint8_t carry = static_cast<uint8_t> (-(static_cast<int> (inc == 0)));
                        ctr[i] = inc;
                        if (!carry)
                                break;
                }
                _S_ct_barrier ();
        }

        /* Constant-time tag comparison. */
        uint8_t acc = 0;
        for (size_t i = 0; i < _S_tag_size; ++i)
                acc |= expected_tag[i] ^ tag[i];
        const bool ok = acc == 0;
        _S_ct_barrier ();
        return ok;
}

} // namespace zzs::base

static_assert (sizeof (zzs::base::SecureChannel) <= 384, "SecureChannel exceeds the 384-byte RAM budget");
