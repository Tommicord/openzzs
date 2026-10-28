/* Tests for the two byte-level training modules: the resumable run-state
 * checkpoint (mllee/train_checkpoint.hpp part A, "MLKC"), the portable weight
 * container (part B, "MLCP") and the fixed-point gradient checker
 * (mllee/train_grad_check.hpp).
 *
 * Reference independence.  Every expectation in this file is recomputed by an
 * independent reference implementation kept in the anonymous namespace below:
 * little-endian loads and stores written by hand (so a writer with the wrong
 * byte order cannot pass by reading its own bytes back through the library's
 * own accessors), a bit-serial CRC-16/CCITT (poly 0x1021, init 0xFFFF) instead
 * of the library's table-driven one, hand-rolled round-half-away division and
 * Q15 ratio, and an integer-only linear+square-error model used as the
 * reference for the gradient checker.  Where the library exposes a documented
 * rule, the test asserts the rule; where a rule is only observable, the test
 * pins the observed value and says so in a comment.
 *
 * The buffers live in std::vector<uint8_t> on the host heap.  The library stays
 * heap-free -- it only ever sees caller-owned memory -- but a parser has to be
 * robust against whatever bytes a hostile or truncated caller hands it, which is
 * exactly what the host build can sweep: every truncation prefix, every single
 * corrupted byte, every documented rejection path. */

#include "mllee/train_checkpoint.hpp"
#include "mllee/train_grad_check.hpp"
#include "mllee/train_loss.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace {

/* ------------------------------------------------------------------ *
 * Byte-level reference helpers                                        *
 * ------------------------------------------------------------------ */

constexpr uint32_t k_meta = _S_ckpt_meta_bytes; /* 32 + 64*16 + 48 = 1104 */

[[nodiscard]] uint32_t
rd32 (const uint8_t *const at) noexcept
{
    return static_cast<uint32_t> (at[0])
           | (static_cast<uint32_t> (at[1]) << 8u)
           | (static_cast<uint32_t> (at[2]) << 16u)
           | (static_cast<uint32_t> (at[3]) << 24u);
}

[[nodiscard]] uint16_t
rd16 (const uint8_t *const at) noexcept
{
    return static_cast<uint16_t> (at[0] | (at[1] << 8u));
}

[[nodiscard]] int16_t
rd16s (const uint8_t *const at) noexcept
{
    return static_cast<int16_t> (rd16 (at));
}

void
wr32 (uint8_t *const at, const uint32_t value) noexcept
{
    at[0] = static_cast<uint8_t> (value & 0xFFu);
    at[1] = static_cast<uint8_t> ((value >> 8u) & 0xFFu);
    at[2] = static_cast<uint8_t> ((value >> 16u) & 0xFFu);
    at[3] = static_cast<uint8_t> ((value >> 24u) & 0xFFu);
}

void
wr16 (uint8_t *const at, const uint16_t value) noexcept
{
    at[0] = static_cast<uint8_t> (value & 0xFFu);
    at[1] = static_cast<uint8_t> ((value >> 8u) & 0xFFu);
}

[[nodiscard]] uint32_t
word_at (const std::vector<uint8_t> &blob, const uint32_t word) noexcept
{
    return rd32 (blob.data () + word * 4u);
}

void
put_word (std::vector<uint8_t> &blob, const uint32_t word,
          const uint32_t value) noexcept
{
    wr32 (blob.data () + word * 4u, value);
}

void
put_byte (std::vector<uint8_t> &blob, const uint32_t offset,
          const uint8_t value) noexcept
{
    blob[offset] = value;
}

/** @brief Bit-serial CRC-16/CCITT, independent of the library's table. */
[[nodiscard]] uint16_t
ref_crc16 (const uint8_t *const data, const uint32_t length,
           uint16_t crc = 0xFFFFu) noexcept
{
    for (uint32_t i = 0u; i < length; ++i)
        {
            crc = static_cast<uint16_t> (crc ^ (static_cast<uint16_t> (data[i]) << 8u));
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc & 0x8000u) != 0u
                          ? static_cast<uint16_t> ((crc << 1u) ^ 0x1021u)
                          : static_cast<uint16_t> (crc << 1u);
        }
    return crc;
}

/** @brief The checkpoint checksum, as documented: [0,28) then [32,total). */
[[nodiscard]] uint16_t
ref_ckpt_crc (const std::vector<uint8_t> &blob, const uint32_t total) noexcept
{
    const uint16_t head = ref_crc16 (blob.data (), 28u);
    return ref_crc16 (blob.data () + 32u, total - 32u, head);
}

/** @brief Rewrite the checksum word so a tampered blob reaches the later rules. */
void
reseal_ckpt (std::vector<uint8_t> &blob, const uint32_t total) noexcept
{
    put_word (blob, 7u, ref_ckpt_crc (blob, total));
}

/** @brief The container's two CRCs, recomputed from the header's own sizes.
 *
 * A corrupted header can claim a payload wider than the buffer; the helper
 * clamps the span it checksums so the test still reaches the reader's size
 * rules instead of reading past the blob.
 */
void
reseal_wck (std::vector<uint8_t> &blob) noexcept
{
    const uint32_t size = static_cast<uint32_t> (blob.size ());
    const uint32_t table_bytes = rd32 (blob.data () + 8u);
    const uint32_t payload_bytes = rd32 (blob.data () + 28u);
    const uint64_t payload_off
        = static_cast<uint64_t> (_S_wck_header_bytes) + table_bytes;
    const uint8_t *payload = blob.data ();
    uint32_t span = 0u;
    if (payload_off < size)
        {
            payload += payload_off;
            const uint64_t available = size - payload_off;
            span = payload_bytes < available
                       ? payload_bytes
                       : static_cast<uint32_t> (available);
        }
    wr16 (blob.data () + 46u, ref_crc16 (payload, span));
    wr16 (blob.data () + 48u, ref_crc16 (blob.data (), 48u));
}

/* ---------------- arithmetic reference for the grad checker ------------- */

/** @brief Round-half-away-from-zero division; zero divisor yields zero. */
[[nodiscard]] int32_t
ref_div_round (int32_t numerator, const int32_t denominator) noexcept
{
    if (denominator == 0)
        return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    if (numerator < 0)
        numerator = -numerator;
    const int32_t den = denominator < 0 ? -denominator : denominator;
    const int32_t quotient = (numerator + den / 2) / den;
    return negative ? -quotient : quotient;
}

[[nodiscard]] int32_t
ref_q15_clamp (const int32_t value) noexcept
{
    if (value > _S_q15_max)
        return _S_q15_max;
    if (value < _S_q15_min)
        return _S_q15_min;
    return value;
}

[[nodiscard]] int32_t
ref_ratio_q15 (const int32_t num, const int32_t den) noexcept
{
    if (den == 0)
        return 0;
    const int64_t scaled = (static_cast<int64_t> (num) << 15) / den;
    return ref_q15_clamp (static_cast<int32_t> (
        scaled > _S_q15_max
            ? _S_q15_max
            : (scaled < -static_cast<int64_t> (_S_q15_one) ? -_S_q15_one
                                                           : scaled)));
}

[[nodiscard]] bool
ref_lane_railed (const int32_t lane) noexcept
{
    return lane >= _S_q15_max || lane <= _S_q15_min;
}

[[nodiscard]] uint32_t
ref_align4 (const uint32_t bytes) noexcept
{
    return (bytes + 3u) & ~3u;
}

/* ------------------------------------------------------------------ *
 * Part A fixture: three tensors, two trainable and one frozen         *
 * ------------------------------------------------------------------ */

struct ThreeParams
{
    std::vector<int8_t> w_code;   /* matrix (3,4): 12 codes, gradients    */
    std::vector<int8_t> b_code;   /* vector (3):   3 codes, gradients     */
    std::vector<int8_t> f_code;   /* vector (4):   4 codes, NO gradients  */
    std::vector<int16_t> w_grad;
    std::vector<int16_t> b_grad;
    std::vector<Trainable> views;

    ThreeParams ()
        : w_code (12u, 0), b_code (3u, 0), f_code (4u, 0), w_grad (12u, 0),
          b_grad (3u, 0)
    {
        views.reserve (3u);
        for (uint32_t i = 0u; i < 12u; ++i)
            {
                w_code[i] = static_cast<int8_t> ((i * 17u) % 200u) - 100;
                w_grad[i] = static_cast<int16_t> (static_cast<int32_t> (i) * 512 - 3000);
            }
        for (uint32_t i = 0u; i < 3u; ++i)
            {
                b_code[i] = static_cast<int8_t> (i * 9u) - 40;
                b_grad[i] = static_cast<int16_t> (-(2000 + i * 97));
            }
        for (uint32_t i = 0u; i < 4u; ++i)
            f_code[i] = static_cast<int8_t> (7 + i);
        views.push_back (Trainable::make (
            w_code.data (), w_grad.data (), Shape::matrix (3u, 4u),
            Flag::RequiresGrad | Flag::Quantized));
        views.push_back (Trainable::make (
            b_code.data (), b_grad.data (), Shape::vector (3u),
            Flag::RequiresGrad | Flag::BiasLike));
        views.push_back (Trainable::make (
            f_code.data (), nullptr, Shape::vector (4u), Flag::Frozen));
        views[0]._M_scale_q15 = 16384;
        views[2]._M_scale_q15 = 12000;
    }

    [[nodiscard]] const Trainable *
    data () const noexcept
    {
        return views.data ();
    }

    [[nodiscard]] uint32_t
    count () const noexcept
    {
        return 3u;
    }

    /** @brief 12 + 3 + 4 codes, 12 + 3 gradient lanes. */
    [[nodiscard]] uint32_t
    elements () const noexcept
    {
        return 19u;
    }

    void
    scratch_codes (const int8_t value) noexcept
    {
        for (int8_t &code : w_code)
            code = value;
        for (int8_t &code : b_code)
            code = value;
        for (int8_t &code : f_code)
            code = value;
    }

    [[nodiscard]] std::vector<int8_t>
    all_codes () const noexcept
    {
        std::vector<int8_t> out;
        out.insert (out.end (), w_code.begin (), w_code.end ());
        out.insert (out.end (), b_code.begin (), b_code.end ());
        out.insert (out.end (), f_code.begin (), f_code.end ());
        return out;
    }

    [[nodiscard]] std::vector<int16_t>
    all_grads () const noexcept
    {
        std::vector<int16_t> out;
        out.insert (out.end (), w_grad.begin (), w_grad.end ());
        out.insert (out.end (), b_grad.begin (), b_grad.end ());
        return out;
    }
};

/** @brief The geometry a bound ThreeParams must produce. */
struct Geometry
{
    uint32_t total = 0u;
    uint32_t grad_bytes = 0u;
};

[[nodiscard]] Geometry
ref_geometry (const ThreeParams &fixture) noexcept
{
    Geometry g;
    g.total = k_meta + ref_align4 (fixture.elements ());
    g.grad_bytes = ref_align4 (24u) + ref_align4 (6u);
    g.total += g.grad_bytes;
    return g;
}

[[nodiscard]] Checkpoint
bind_ckpt (const ThreeParams &fixture) noexcept
{
    Checkpoint ckpt;
    REQUIRE (status_ok (ckpt.bind (fixture.data (), fixture.count ())));
    return ckpt;
}

[[nodiscard]] RunState
sample_run () noexcept
{
    RunState run;
    run._M_step = 41u;
    run._M_loss_scale_q15 = 16384;
    run._M_clean_steps = 7u;
    run._M_best_loss_q7 = -1234;
    run._M_best_step = 33u;
    run._M_epochs = 2u;
    run._M_first_loss_q7 = 9000;
    run._M_last_loss_q7 = -55;
    run._M_saturation_total = 8u;
    run._M_scale_halves = 3u;
    run._M_scale_grows = 1u;
    run._M_have_best = 1u;
    return run;
}

/** @brief Serialize a bound fixture into a freshly sized buffer. */
[[nodiscard]] std::vector<uint8_t>
write_ckpt (const ThreeParams &fixture, const RunState &run,
            uint32_t &out_bytes) noexcept
{
    const Geometry geometry = ref_geometry (fixture);
    std::vector<uint8_t> blob (geometry.total, 0u);
    const Checkpoint ckpt = bind_ckpt (fixture);
    const Status status = ckpt.serialize (blob.data (), geometry.total, run,
                                         out_bytes);
    REQUIRE (status_ok (status));
    return blob;
}

/** @brief The expected rejection for a truncated checkpoint, per rule order. */
[[nodiscard]] Status
expected_truncation (const uint32_t length, const uint32_t total) noexcept
{
    if (length < k_meta)
        return Status::TRUNCATED;
    if (length < total)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

} // namespace

/* ==================================================================== *
 * Part A -- resumable run-state checkpoint ("MLKC")                     *
 * ==================================================================== */

TEST_CASE ("checkpoint wire constants are the documented fixed layout")
{
    SECTION ("header, entry, run-state and info sizes")
        {
            static_assert (sizeof (CheckpointHeader) == 32u, "header wire size");
            static_assert (sizeof (CheckpointEntry) == 16u, "entry wire size");
            static_assert (sizeof (RunState) == 48u, "run wire size");
            static_assert (sizeof (CheckpointInfo) == 28u, "info size");
            REQUIRE (k_meta == 1104u);
            REQUIRE (_S_ckpt_header_bytes == 32u);
            REQUIRE (_S_ckpt_entry_words == 4u);
            REQUIRE (_S_ckpt_run_words == 12u);
            REQUIRE (_S_ckpt_run_words * 4u == sizeof (RunState));
            REQUIRE (_S_max_trainer_params * _S_ckpt_entry_words * 4u == 1024u);
        }

    SECTION ("magic and version spell MLKC little-endian")
        {
            REQUIRE (_S_ckpt_magic == 0x434B4C4Du);
            REQUIRE (static_cast<uint8_t> (_S_ckpt_magic & 0xFFu)
                     == static_cast<uint8_t> ('M'));
            REQUIRE (static_cast<uint8_t> ((_S_ckpt_magic >> 8u) & 0xFFu)
                     == static_cast<uint8_t> ('L'));
            REQUIRE (static_cast<uint8_t> ((_S_ckpt_magic >> 16u) & 0xFFu)
                     == static_cast<uint8_t> ('K'));
            REQUIRE (static_cast<uint8_t> ((_S_ckpt_magic >> 24u) & 0xFFu)
                     == static_cast<uint8_t> ('C'));
            REQUIRE (_S_ckpt_version == 1u);
        }

    SECTION ("checkpoint_bytes sizes the meta prefix plus both aligned streams")
        {
            REQUIRE (checkpoint_bytes (0u, 0u) == k_meta);
            REQUIRE (checkpoint_bytes (1u, 1u) == k_meta + 4u + 4u);
            REQUIRE (checkpoint_bytes (3u, 19u) == k_meta + 20u + 40u);
            REQUIRE (checkpoint_bytes (3u, 20u) == k_meta + 20u + 40u);
            /* A parameter count the serializer rejects saturates the query
               instead of wrapping, so a caller can never under-allocate. */
            REQUIRE (checkpoint_bytes (_S_max_trainer_params + 1u, 1u)
                     == 0xFFFFFFFFu);
            REQUIRE (checkpoint_bytes (0xFFFFFFFFu, 0xFFFFFFFFu) == 0xFFFFFFFFu);
            REQUIRE (checkpoint_bytes (_S_max_trainer_params, 0u) == k_meta);
        }
}

TEST_CASE ("checkpoint bind validates the parameter list and precomputes layout")
{
    const ThreeParams fixture;

    SECTION ("a bound list reports the exact geometry")
        {
            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (fixture.data (), fixture.count ())));
            const CheckpointInfo info = ckpt.info ();
            REQUIRE (info._M_entry_count == 3u);
            REQUIRE (info._M_element_total == 19u);
            REQUIRE (info._M_param_off == k_meta);
            REQUIRE (info._M_param_bytes == 20u);
            REQUIRE (info._M_grad_off == k_meta + 20u);
            /* the frozen tensor contributes no lane stream, so the bound
               geometry is below the all-trainable checkpoint_bytes query */
            REQUIRE (info._M_grad_bytes == 32u);
            REQUIRE (info._M_total_bytes == ref_geometry (fixture).total);
            REQUIRE (info._M_total_bytes < checkpoint_bytes (3u, 19u));
            static_assert (sizeof (CheckpointInfo) == 28u, "seven words");
        }

    SECTION ("null and empty lists are BAD_LAYOUT")
        {
            Checkpoint ckpt;
            REQUIRE (ckpt.bind (nullptr, 1u) == Status::BAD_LAYOUT);
            REQUIRE (ckpt.bind (fixture.data (), 0u) == Status::BAD_LAYOUT);
        }

    SECTION ("one entry is legal")
        {
            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (fixture.data (), 1u)));
            REQUIRE (ckpt.info ()._M_entry_count == 1u);
            REQUIRE (ckpt.info ()._M_element_total == 12u);
            REQUIRE (ckpt.info ()._M_grad_bytes == 24u);
        }

    SECTION ("the format maximum is 64 entries, one more is UNSUPPORTED")
        {
            std::vector<int8_t> code (4u, 3);
            std::vector<int16_t> grad (4u, 0);
            const Trainable one
                = Trainable::make (code.data (), grad.data (),
                                   Shape::vector (4u), Flag::RequiresGrad);
            std::vector<Trainable> many (64u, one);
            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (many.data (), 64u)));
            REQUIRE (ckpt.info ()._M_element_total == 256u);
            REQUIRE (ckpt.info ()._M_param_bytes == 256u);
            REQUIRE (ckpt.info ()._M_grad_bytes == 512u);
            REQUIRE (ckpt.info ()._M_total_bytes == checkpoint_bytes (64u, 256u));
            many.push_back (one);
            REQUIRE (ckpt.bind (many.data (), 65u) == Status::UNSUPPORTED);
        }

    SECTION ("a malformed view propagates its own validation status")
        {
            ThreeParams bad;
            bad.views[1]._M_count = 4u; /* disagrees with vector(3) */
            Checkpoint ckpt;
            REQUIRE (ckpt.bind (bad.data (), 3u) == Status::BAD_LAYOUT);
            /* bind refuses before publishing anything */
            REQUIRE (ckpt.info ()._M_entry_count == 0u);
            REQUIRE (ckpt.info ()._M_total_bytes == 0u);
        }

    SECTION ("an element total past _S_max_elements is OUT_OF_RANGE")
        {
            std::vector<int8_t> code (65535u, 5);
            const Trainable big = Trainable::make (code.data (), nullptr,
                                                   Shape::vector (65535u), 0u);
            REQUIRE (status_ok (big.validate ()));
            std::vector<Trainable> two (2u, big);
            Checkpoint ckpt;
            REQUIRE (ckpt.bind (two.data (), 2u) == Status::OUT_OF_RANGE);
            REQUIRE (status_ok (ckpt.bind (two.data (), 1u)));
        }

    SECTION ("the fixture streams really are SIMD-ready")
        {
            for (const Trainable &param : fixture.views)
                REQUIRE (param.simd_ready ());
            REQUIRE (simd_aligned (fixture.data ()));
        }
}

TEST_CASE ("checkpoint serialize/deserialize round trip preserves every byte")
{
    ThreeParams fixture;
    const Geometry geometry = ref_geometry (fixture);
    const RunState run = sample_run ();
    uint32_t written = 0u;
    const std::vector<uint8_t> blob = write_ckpt (fixture, run, written);

    SECTION ("out_bytes is the exact serialized length")
        {
            REQUIRE (written == geometry.total);
            REQUIRE (blob.size () == geometry.total);
        }

    SECTION ("header words carry the geometry, the checksum word is a CRC-16")
        {
            REQUIRE (word_at (blob, 0u) == _S_ckpt_magic);
            REQUIRE (word_at (blob, 1u) == _S_ckpt_version);
            REQUIRE (word_at (blob, 2u) == _S_ckpt_header_bytes);
            REQUIRE (word_at (blob, 3u) == fixture.count ());
            REQUIRE (word_at (blob, 4u) == fixture.elements ());
            REQUIRE (word_at (blob, 5u) == geometry.total);
            REQUIRE (word_at (blob, 6u) == 0u);
            REQUIRE (word_at (blob, 7u) <= 0xFFFFu);
            REQUIRE (static_cast<uint16_t> (word_at (blob, 7u))
                     == ref_ckpt_crc (blob, geometry.total));
            REQUIRE (static_cast<uint16_t> (word_at (blob, 7u))
                     == Checkpoint::checksum (blob.data (), geometry.total));
        }

    SECTION ("entries are laid out in caller order with explicit offsets")
        {
            const uint32_t code_off = k_meta;
            REQUIRE (word_at (blob, 8u) == code_off);
            REQUIRE (word_at (blob, 9u) == code_off + 20u);
            REQUIRE (word_at (blob, 10u) == 12u);
            REQUIRE (word_at (blob, 11u) == 12u);
            REQUIRE (word_at (blob, 12u) == code_off + 12u);
            REQUIRE (word_at (blob, 13u) == code_off + 20u + 24u);
            REQUIRE (word_at (blob, 14u) == 3u);
            REQUIRE (word_at (blob, 15u) == 3u);
            REQUIRE (word_at (blob, 16u) == code_off + 15u);
            REQUIRE (word_at (blob, 17u) == 0u); /* frozen: no lane stream */
            REQUIRE (word_at (blob, 18u) == 4u);
            REQUIRE (word_at (blob, 19u) == 4u);
        }

    SECTION ("the code stream is stride-1 and the pad bytes are 0xFF")
        {
            for (uint32_t i = 0u; i < 19u; ++i)
                REQUIRE (blob[k_meta + i]
                         == static_cast<uint8_t> (fixture.all_codes ()[i]));
            REQUIRE (blob[k_meta + 19u] == 0xFFu);
            REQUIRE (blob[k_meta + 20u - 1u] == 0xFFu);
            for (uint32_t pad = k_meta + 19u; pad < k_meta + 20u; ++pad)
                REQUIRE (blob[pad] == 0xFFu);
        }

    SECTION ("gradient lanes are little-endian int16")
        {
            const uint32_t grad_off = k_meta + 20u;
            const std::vector<int16_t> grads = fixture.all_grads ();
            for (uint32_t i = 0u; i < grads.size (); ++i)
                REQUIRE (rd16s (blob.data () + grad_off + 2u * i) == grads[i]);
            /* the bias stream is 6 bytes then 2 pad bytes before the end */
            REQUIRE (grad_off + 30u == geometry.total - 2u);
        }

    SECTION ("a reload into a fresh fixture is byte-identical")
        {
            ThreeParams other;
            other.scratch_codes (17);
            for (uint32_t i = 0u; i < 12u; ++i)
                other.w_grad[i] = static_cast<int16_t> (i);
            for (uint32_t i = 0u; i < 3u; ++i)
                other.b_grad[i] = static_cast<int16_t> (-i);
            Checkpoint ckpt = bind_ckpt (other);
            RunState loaded;
            REQUIRE (status_ok (ckpt.deserialize (blob.data (), written, loaded)));
            REQUIRE (other.w_code == fixture.w_code);
            REQUIRE (other.b_code == fixture.b_code);
            REQUIRE (other.f_code == fixture.f_code);
            REQUIRE (other.w_grad == fixture.w_grad);
            REQUIRE (other.b_grad == fixture.b_grad);
            REQUIRE (loaded._M_step == run._M_step);
        }

    SECTION ("deserialize never touches the streams on a rejected buffer")
        {
            ThreeParams other;
            const std::vector<int8_t> before = other.all_codes ();
            const std::vector<int16_t> lanes_before = other.all_grads ();
            Checkpoint ckpt = bind_ckpt (other);
            RunState loaded;
            REQUIRE (not status_ok (ckpt.deserialize (
                blob.data (), written - 1u, loaded)));
            REQUIRE (other.all_codes () == before);
            REQUIRE (other.all_grads () == lanes_before);
        }

    SECTION ("codes in [-127,127]: the -128 rail never survives a round trip")
        {
            for (uint32_t i = 0u; i < 19u; ++i)
                {
                    const int8_t code = fixture.all_codes ()[i];
                    REQUIRE (code >= -127);
                    REQUIRE (code <= 127);
                    REQUIRE (code != -128);
                }
        }
}

TEST_CASE ("checkpoint serialize rejects malformed calls before writing")
{
    ThreeParams fixture;
    const Geometry geometry = ref_geometry (fixture);
    const RunState run = sample_run ();

    SECTION ("an unbound checkpoint cannot serialize")
        {
            Checkpoint ckpt;
            std::vector<uint8_t> blob (geometry.total, 0u);
            uint32_t written = 0u;
            REQUIRE (ckpt.serialize (blob.data (), geometry.total, run, written)
                    == Status::BAD_LAYOUT);
            REQUIRE (written == 0u);
        }

    SECTION ("null and misaligned destinations are BAD_LAYOUT")
        {
            Checkpoint ckpt = bind_ckpt (fixture);
            std::vector<uint8_t> blob (geometry.total, 0u);
            uint32_t written = 0u;
            REQUIRE (ckpt.serialize (nullptr, geometry.total, run, written)
                    == Status::BAD_LAYOUT);
            REQUIRE (ckpt.serialize (blob.data () + 1u, geometry.total - 1u, run,
                                    written)
                     == Status::BAD_LAYOUT);
            REQUIRE (blob[0] == 0u);
        }

    SECTION ("a short buffer is ARENA_OVERFLOW, an exact one is OK")
        {
            Checkpoint ckpt = bind_ckpt (fixture);
            std::vector<uint8_t> blob (geometry.total + 8u, 0u);
            uint32_t written = 0u;
            REQUIRE (ckpt.serialize (blob.data (), geometry.total - 1u, run,
                                    written)
                     == Status::ARENA_OVERFLOW);
            REQUIRE (written == 0u);
            REQUIRE (status_ok (ckpt.serialize (blob.data (), geometry.total, run,
                                                written)));
            REQUIRE (written == geometry.total);
        }

    SECTION ("a -128 code is rejected and nothing is written")
        {
            fixture.views[0]._M_code[5] = -128;
            Checkpoint ckpt = bind_ckpt (fixture);
            std::vector<uint8_t> blob (geometry.total, 0xA5u);
            uint32_t written = 0u;
            REQUIRE (ckpt.serialize (blob.data (), geometry.total, run, written)
                    == Status::BAD_LAYOUT);
            REQUIRE (written == 0u);
            for (uint8_t byte : blob)
                REQUIRE (byte == 0xA5u);
        }

    SECTION ("a -128 code in any tensor blocks the whole checkpoint")
        {
            for (uint32_t tensor = 0u; tensor < 3u; ++tensor)
                {
                    ThreeParams local;
                    local.views[tensor]._M_code[0] = -128;
                    Checkpoint ckpt;
                    REQUIRE (status_ok (ckpt.bind (local.data (), 3u)));
                    std::vector<uint8_t> blob (geometry.total, 0u);
                    uint32_t written = 0u;
                    REQUIRE (ckpt.serialize (blob.data (), geometry.total, run,
                                            written)
                             == Status::BAD_LAYOUT);
                }
        }
}

TEST_CASE ("checkpoint inspect reports header defects in the documented order")
{
    ThreeParams fixture;
    const Geometry geometry = ref_geometry (fixture);
    uint32_t written = 0u;
    std::vector<uint8_t> blob = write_ckpt (fixture, sample_run (), written);
    CheckpointInfo info;
    RunState run;

    const auto probe = [&] (std::vector<uint8_t> &buf) {
        return Checkpoint::inspect (buf.data (), static_cast<uint32_t> (buf.size ()),
                                    fixture.data (), fixture.count (), info, run);
    };

    SECTION ("a clean blob validates against its own destination list")
        {
            REQUIRE (status_ok (probe (blob)));
            REQUIRE (info._M_total_bytes == geometry.total);
            REQUIRE (info._M_grad_bytes == 32u);
            REQUIRE (info._M_element_total == 19u);
        }

    SECTION ("null buffer is BAD_LAYOUT, a short one TRUNCATED")
        {
            REQUIRE (Checkpoint::inspect (nullptr, 4096u, nullptr, 0u, info, run)
                     == Status::BAD_LAYOUT);
            REQUIRE (Checkpoint::inspect (blob.data (), k_meta - 1u, nullptr, 0u,
                                         info, run)
                     == Status::TRUNCATED);
            REQUIRE (Checkpoint::inspect (blob.data (), 0u, nullptr, 0u, info, run)
                     == Status::TRUNCATED);
        }

    SECTION ("a misaligned buffer is BAD_LAYOUT")
        {
            REQUIRE (Checkpoint::inspect (blob.data () + 1u, written - 1u, nullptr,
                                         0u, info, run)
                     == Status::BAD_LAYOUT);
            REQUIRE (Checkpoint::inspect (blob.data () + 2u, written - 2u, nullptr,
                                         0u, info, run)
                     == Status::BAD_LAYOUT);
        }

    SECTION ("magic, version, header size and the reserved word each have a status")
        {
            put_word (blob, 0u, _S_ckpt_magic ^ 0x01000000u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_MAGIC);

            put_word (blob, 0u, _S_ckpt_magic);
            put_word (blob, 1u, 2u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_VERSION);

            put_word (blob, 1u, _S_ckpt_version);
            put_word (blob, 2u, _S_ckpt_header_bytes + 4u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 2u, _S_ckpt_header_bytes);
            put_word (blob, 6u, 1u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 6u, 0u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (status_ok (probe (blob)));
        }

    SECTION ("the size words must agree with the buffer and with each other")
        {
            put_word (blob, 5u, geometry.total + 4u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 5u, k_meta - 4u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 5u, geometry.total);
            put_word (blob, 3u, _S_max_trainer_params + 1u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 3u, fixture.count ());
            put_word (blob, 4u, 0u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);

            put_word (blob, 4u, _S_max_elements + 1u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("an entry-count/header-total disagreement is BAD_LAYOUT")
        {
            /* the destination list is what pins the count; a blob whose count
               word lies is caught before the CRC even matters only if the
               streams still tile, so reseal to reach the count rule */
            put_word (blob, 3u, 2u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT); /* idempotent */
        }
}

TEST_CASE ("checkpoint stream rules reject relocation, zero counts and padding lies")
{
    ThreeParams fixture;
    const Geometry geometry = ref_geometry (fixture);
    uint32_t written = 0u;
    std::vector<uint8_t> blob = write_ckpt (fixture, sample_run (), written);
    CheckpointInfo info;
    RunState run;

    const auto probe = [&] (std::vector<uint8_t> &buf) {
        return Checkpoint::inspect (buf.data (), static_cast<uint32_t> (buf.size ()),
                                    fixture.data (), fixture.count (), info, run);
    };
    const auto probe_no_dest = [&] (std::vector<uint8_t> &buf) {
        return Checkpoint::inspect (buf.data (),
                                    static_cast<uint32_t> (buf.size ()), nullptr, 0u,
                                    info, run);
    };

    SECTION ("a zero element count is BAD_LAYOUT")
        {
            put_word (blob, 10u, 0u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("a shape/count cross-check word that lies is BAD_LAYOUT")
        {
            put_word (blob, 11u, 11u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
            REQUIRE (probe (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("a reordered code stream is corruption, not a permutation")
        {
            const uint32_t first_code = word_at (blob, 8u);
            const uint32_t second_code = word_at (blob, 12u);
            put_word (blob, 8u, second_code);
            put_word (blob, 12u, first_code);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("a relocated lane stream is BAD_LAYOUT")
        {
            put_word (blob, 9u, word_at (blob, 9u) + 4u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("a lane stream offset that is not 4-aligned is BAD_LAYOUT")
        {
            put_word (blob, 9u, k_meta + 20u + 1u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("a lane stream that claims to precede the code stream is rejected")
        {
            put_word (blob, 9u, k_meta - 4u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("an entry offset that runs past the blob is BAD_LAYOUT")
        {
            put_word (blob, 16u, geometry.total - 2u); /* 4 codes cannot fit */
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }

    SECTION ("the element total must equal the sum of entry counts")
        {
            put_word (blob, 4u, 18u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
            /* the aligned span then disagrees with the total_bytes word too */
            put_word (blob, 4u, 19u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (status_ok (probe_no_dest (blob)));
        }

    SECTION ("a blob with lanes the header cannot account for is BAD_LAYOUT")
        {
            /* give the frozen tensor a lane offset: the expected total moves */
            put_word (blob, 17u, k_meta + 20u + 30u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_no_dest (blob) == Status::BAD_LAYOUT);
        }
}

TEST_CASE ("checkpoint destination rules and the CRC trust order")
{
    ThreeParams fixture;
    const Geometry geometry = ref_geometry (fixture);
    uint32_t written = 0u;
    std::vector<uint8_t> blob = write_ckpt (fixture, sample_run (), written);
    CheckpointInfo info;
    RunState run;

    const auto probe_with = [&] (const Trainable *params, uint32_t count) {
        return Checkpoint::inspect (blob.data (),
                                    static_cast<uint32_t> (blob.size ()), params,
                                    count, info, run);
    };

    SECTION ("CRC failure precedes every data rule")
        {
            /* flip one code byte: structural words still agree, so the only
               thing that can catch it is the checksum */
            put_byte (blob, k_meta + 3u, static_cast<uint8_t> (0x00));
            REQUIRE (probe_with (nullptr, 0u) == Status::CORRUPT_STREAM);
            REQUIRE (probe_with (fixture.data (), fixture.count ())
                     == Status::CORRUPT_STREAM);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_with (nullptr, 0u) == Status::OK);
        }

    SECTION ("a -128 code in the stream is CORRUPT_STREAM, data not structure")
        {
            put_byte (blob, k_meta + 4u, 0x80u);
            reseal_ckpt (blob, geometry.total);
            REQUIRE (probe_with (nullptr, 0u) == Status::OK);
            REQUIRE (probe_with (fixture.data (), fixture.count ())
                     == Status::CORRUPT_STREAM);
            REQUIRE (probe_with (fixture.data (), 1u) == Status::BAD_LAYOUT);
        }

    SECTION ("a destination list that does not match the entries is rejected")
        {
            REQUIRE (probe_with (fixture.data (), 2u) == Status::BAD_LAYOUT);
            REQUIRE (probe_with (fixture.data (), 1u) == Status::BAD_LAYOUT);
            REQUIRE (probe_with (nullptr, 3u) == Status::BAD_LAYOUT);
            ThreeParams other;
            other.views[0]._M_grad = nullptr; /* grad presence must agree */
            REQUIRE (Checkpoint::inspect (blob.data (),
                                          static_cast<uint32_t> (blob.size ()),
                                          other.data (), 3u, info, run)
                     == Status::BAD_LAYOUT);
            other.views[0]._M_grad = fixture.w_grad.data ();
            other.views[2]._M_code[0] = -128; /* irrelevant: data check is on
                                                 the blob, not the view */
            REQUIRE (status_ok (Checkpoint::inspect (
                blob.data (), static_cast<uint32_t> (blob.size ()), other.data (),
                3u, info, run)));
        }

    SECTION ("a misaligned destination stream is BAD_LAYOUT")
        {
            std::vector<int8_t> code (12u, 1);
            std::vector<int16_t> grad (12u, 0);
            const Trainable misaligned
                = Trainable::make (code.data () + 1u, grad.data (),
                                   Shape::matrix (3u, 4u), Flag::RequiresGrad);
            REQUIRE (not misaligned.simd_ready ());
            ThreeParams local;
            std::vector<Trainable> list;
            list.reserve (3u);
            list.push_back (misaligned);
            list.push_back (local.views[1]);
            list.push_back (local.views[2]);
            REQUIRE (Checkpoint::inspect (blob.data (),
                                          static_cast<uint32_t> (blob.size ()),
                                          list.data (), 3u, info, run)
                     == Status::BAD_LAYOUT);
        }

    SECTION ("an invalid destination view is BAD_LAYOUT")
        {
            ThreeParams local;
            local.views[1]._M_code = nullptr;
            REQUIRE (Checkpoint::inspect (blob.data (),
                                          static_cast<uint32_t> (blob.size ()),
                                          local.data (), 3u, info, run)
                     == Status::BAD_LAYOUT);
        }

    SECTION ("the checksum function is the documented fold, verified independently")
        {
            const uint16_t library = Checkpoint::checksum (blob.data (),
                                                          geometry.total);
            REQUIRE (library == ref_ckpt_crc (blob, geometry.total));
            /* the fold identity the header documents: 28 bytes, skip the
               checksum word, then the payload */
            uint16_t fold = ref_crc16 (blob.data (), 28u);
            fold = ref_crc16 (blob.data () + 32u, geometry.total - 32u, fold);
            REQUIRE (fold == library);
            REQUIRE (checkpoint_crc_fold (0xFFFFu, blob.data (), 1u)
                     == ref_crc16 (blob.data (), 1u));
            REQUIRE (Checkpoint::checksum (blob.data (), geometry.total)
                     != Checkpoint::checksum (blob.data (), geometry.total - 4u));
        }

    SECTION ("run state words survive a full round trip exactly")
        {
            const RunState saved = sample_run ();
            REQUIRE (status_ok (probe_with (fixture.data (), fixture.count ())));
            REQUIRE (run._M_step == saved._M_step);
            REQUIRE (run._M_loss_scale_q15 == saved._M_loss_scale_q15);
            REQUIRE (run._M_clean_steps == saved._M_clean_steps);
            REQUIRE (run._M_best_loss_q7 == saved._M_best_loss_q7);
            REQUIRE (run._M_best_step == saved._M_best_step);
            REQUIRE (run._M_epochs == saved._M_epochs);
            REQUIRE (run._M_first_loss_q7 == saved._M_first_loss_q7);
            REQUIRE (run._M_last_loss_q7 == saved._M_last_loss_q7);
            REQUIRE (run._M_saturation_total == saved._M_saturation_total);
            REQUIRE (run._M_scale_halves == saved._M_scale_halves);
            REQUIRE (run._M_scale_grows == saved._M_scale_grows);
            REQUIRE (run._M_have_best == saved._M_have_best);
            /* and they are stored where the format says, at word 8+64*4 */
            const uint32_t base = 8u + _S_max_trainer_params * _S_ckpt_entry_words;
            REQUIRE (base == 264u);
            REQUIRE (word_at (blob, base) == saved._M_step);
            REQUIRE (word_at (blob, base + 1u)
                     == static_cast<uint32_t> (saved._M_loss_scale_q15));
            REQUIRE (rd32 (blob.data () + base * 4u + 3u * 4u)
                     == static_cast<uint32_t> (saved._M_best_loss_q7));
        }

    SECTION ("every RunState bit pattern round-trips, negatives included")
        {
            const int32_t extremes[4] = { 0, -1, _S_q15_min, 0x7FFFFFFF };
            for (int32_t a : extremes)
                {
                    for (uint32_t b : { 0u, 1u, 0xFFFFFFFFu, 264u })
                        {
                            RunState run_out;
                            run_out._M_step = b;
                            run_out._M_loss_scale_q15 = a;
                            run_out._M_first_loss_q7 = a;
                            run_out._M_best_loss_q7 = b == 0u ? a : -a;
                            run_out._M_have_best = b % 2u;
                            std::vector<uint8_t> buf (geometry.total, 0u);
                            Checkpoint ckpt = bind_ckpt (fixture);
                            uint32_t written_local = 0u;
                            REQUIRE (status_ok (ckpt.serialize (
                                buf.data (), geometry.total, run_out,
                                written_local)));
                            RunState back;
                            REQUIRE (status_ok (ckpt.deserialize (
                                buf.data (), written_local, back)));
                            REQUIRE (back._M_step == run_out._M_step);
                            REQUIRE (back._M_loss_scale_q15
                                     == run_out._M_loss_scale_q15);
                            REQUIRE (back._M_first_loss_q7
                                     == run_out._M_first_loss_q7);
                            REQUIRE (back._M_best_loss_q7
                                     == run_out._M_best_loss_q7);
                            REQUIRE (back._M_have_best == run_out._M_have_best);
                        }
                }
        }
}

TEST_CASE ("checkpoint truncation sweep rejects every short prefix")
{
    ThreeParams fixture;
    uint32_t written = 0u;
    const std::vector<uint8_t> blob = write_ckpt (fixture, sample_run (), written);
    REQUIRE (blob.size () == written);

    /* The highest-value test in the file: a reader must never read past the
       bytes it was given, and it must name the reason it stopped.  Every
       prefix length below the full size is probed, both with and without a
       destination list. */
    uint32_t truncated = 0u;
    uint32_t layout = 0u;
    for (uint32_t length = 0u; length <= written; ++length)
        {
            CheckpointInfo info;
            RunState run;
            const Status with_dest = Checkpoint::inspect (
                blob.data (), length, fixture.data (), fixture.count (), info, run);
            CheckpointInfo info2;
            RunState run2;
            const Status no_dest = Checkpoint::inspect (
                blob.data (), length, nullptr, 0u, info2, run2);
            const Status want = expected_truncation (length, written);
            REQUIRE (with_dest == want);
            REQUIRE (no_dest == want);
            if (want == Status::TRUNCATED)
                ++truncated;
            if (want == Status::BAD_LAYOUT)
                ++layout;
            if (status_ok (want))
                {
                    REQUIRE (info._M_total_bytes == written);
                    REQUIRE (info2._M_total_bytes == written);
                }
        }
    REQUIRE (truncated == k_meta);
    REQUIRE (layout == written - k_meta);

    SECTION ("a deserialize into a short buffer leaves the tensors alone")
        {
            ThreeParams other;
            const std::vector<int8_t> before = other.all_codes ();
            Checkpoint ckpt = bind_ckpt (other);
            for (uint32_t length = 0u; length < written; ++length)
                {
                    RunState run;
                    REQUIRE (not status_ok (ckpt.deserialize (blob.data (),
                                                              length, run)));
                }
            REQUIRE (other.all_codes () == before);
        }
}

TEST_CASE ("checkpoint single-byte corruption sweep")
{
    ThreeParams fixture;
    uint32_t written = 0u;
    const std::vector<uint8_t> base = write_ckpt (fixture, sample_run (), written);

    /* Everything from the meta prefix on is payload as far as the CRC is
       concerned (bytes [0,28) and [32,total)), and the header's own words are
       checked before it, so a one-byte flip inside the streams must always
       surface as CORRUPT_STREAM rather than a silent mis-read. */
    uint32_t probes = 0u;
    for (uint32_t offset = k_meta; offset < written; ++offset)
        {
            std::vector<uint8_t> buf = base;
            buf[offset] = static_cast<uint8_t> (buf[offset] ^ 0x01u);
            CheckpointInfo info;
            RunState run;
            const Status status = Checkpoint::inspect (buf.data (), written,
                                                      nullptr, 0u, info, run);
            REQUIRE (status == Status::CORRUPT_STREAM);
            ++probes;
        }
    REQUIRE (probes == written - k_meta);

    SECTION ("a flip in the checksum word's low half is caught, its pad is not")
        {
            /* inspect compares only the low 16 bits of word 7, so the two pad
               bytes above the CRC are outside the trust chain: recorded here so
               the format's real coverage is stated in the test, not assumed. */
            for (uint32_t offset = 28u; offset < 30u; ++offset)
                {
                    std::vector<uint8_t> buf = base;
                    buf[offset] = static_cast<uint8_t> (buf[offset] ^ 0x80u);
                    CheckpointInfo info;
                    RunState run;
                    REQUIRE (Checkpoint::inspect (buf.data (), written, nullptr, 0u,
                                                 info, run)
                             == Status::CORRUPT_STREAM);
                }
            for (uint32_t offset = 30u; offset < 32u; ++offset)
                {
                    std::vector<uint8_t> buf = base;
                    buf[offset] = static_cast<uint8_t> (buf[offset] ^ 0x80u);
                    CheckpointInfo info;
                    RunState run;
                    REQUIRE (Checkpoint::inspect (buf.data (), written, nullptr, 0u,
                                                 info, run)
                             == Status::OK);
                    REQUIRE (word_at (buf, 7u) > 0xFFFFu);
                }
        }

    SECTION ("resealing restores acceptance, so the sweep really tests the CRC")
        {
            std::vector<uint8_t> buf = base;
            buf[k_meta] = static_cast<uint8_t> (buf[k_meta] ^ 0x40u);
            CheckpointInfo info;
            RunState run;
            REQUIRE (Checkpoint::inspect (buf.data (), written, nullptr, 0u, info,
                                         run)
                     == Status::CORRUPT_STREAM);
            reseal_ckpt (buf, written);
            REQUIRE (Checkpoint::inspect (buf.data (), written, nullptr, 0u, info,
                                         run)
                     == Status::OK);
        }
}

namespace {

/* ------------------------------------------------------------------ *
 * Deterministic toy trainer: one vector(4) parameter, integer model    *
 * ------------------------------------------------------------------ */

constexpr int32_t k_traj_x[4] = { 1, 2, 1, 1 };
constexpr int32_t k_traj_target = 9;
constexpr int32_t k_traj_lr_divisor = 2048;

[[nodiscard]] int32_t
ref_acc (const int8_t *const code) noexcept
{
    int32_t acc = 0;
    for (uint32_t i = 0u; i < 4u; ++i)
        acc += k_traj_x[i] * code[i];
    return acc;
}

[[nodiscard]] int8_t
ref_clamp_code (const int32_t value) noexcept
{
    if (value > 127)
        return 127;
    if (value < -127)
        return -127;
    return static_cast<int8_t> (value);
}

/** @brief The toy's Q7 loss: squared residual plus a per-step log term. */
[[nodiscard]] int32_t
ref_traj_loss (const int8_t *const code, const uint32_t step) noexcept
{
    const int32_t err = ref_acc (code) - k_traj_target;
    return err * err + static_cast<int32_t> (step) * 3;
}

/** @brief The toy's analytic lane domain: 256 Q7 units per unit real. */
[[nodiscard]] int32_t
ref_traj_lane (const int8_t *const code, const uint32_t index) noexcept
{
    const int32_t err = ref_acc (code) - k_traj_target;
    return 256 * 2 * err * k_traj_x[index];
}

/** @brief One reference step: lanes from the codes as they were at the start of
 *         the step (a backward pass reads, an update writes: never interleaved),
 *         then the SGD-style code update (truncating division). */
void
ref_traj_step (std::vector<int8_t> &code, uint32_t step, int32_t &loss_out) noexcept
{
    loss_out = ref_traj_loss (code.data (), step);
    const int32_t err = ref_acc (code.data ()) - k_traj_target;
    for (uint32_t i = 0u; i < 4u; ++i)
        {
            const int32_t lane = 256 * 2 * err * k_traj_x[i];
            const int32_t lane16 = lane > 32767 ? 32767
                                                 : (lane < -32768 ? -32768 : lane);
            code[i] = ref_clamp_code (code[i] - lane16 / k_traj_lr_divisor);
        }
}

Status
traj_backward (const Trainable &output, const Trainable &grad_output,
               GradContext &context) noexcept
{
    (void) output;
    if (grad_output.grad_at (0) == 0)
        return Status::INTERNAL;
    const Trainable &param = context._M_operands[0];
    /* The toy is deliberately scale-free: the lanes depend only on the codes,
       so a restored run reproduces its partner bit for bit regardless of the
       loss scale it resumes at. */
    for (uint32_t i = 0u; i < param._M_count; ++i)
        param.set_grad (i, ref_traj_lane (param._M_code, i));
    return Status::OK;
}

Status
traj_forward (StepContext &context) noexcept
{
    const Trainable &param = context._M_params[0];
    context._M_loss_q7 = ref_traj_loss (param._M_code, context._M_step);
    context._M_loss.set_code (0, context._M_loss_q7);
    if (!context._M_tape->is_open ())
        return Status::OK;
    const Trainable operands[1] = { param };
    return context._M_tape->record (traj_backward, 11u, context._M_loss, operands,
                                   1u, nullptr, 0u);
}

Status
traj_update (StepContext &context) noexcept
{
    const Trainable &param = context._M_params[0];
    for (uint32_t i = 0u; i < param._M_count; ++i)
        param.set_code (i, param.code_at (i)
                              - param.grad_at (i) / k_traj_lr_divisor);
    return Status::OK;
}

/** @brief The forward that fails, to prove a resume keeps the run's counters. */
Status
traj_forward_fails (StepContext &context) noexcept
{
    (void) context;
    return Status::OUT_OF_RANGE;
}

struct TrajFixture
{
    alignas (8) uint8_t pool_backing[256] = { 0u };
    alignas (8) Node tape_nodes[4] = {};
    std::vector<int8_t> code;
    std::vector<int16_t> grad;
    std::vector<int8_t> loss_code;
    std::vector<int16_t> loss_grad;
    std::vector<Trainable> views;
    Pool pool;
    Tape tape;
    StepContext ctx;

    TrajFixture ()
        : pool_backing{}, tape_nodes{}, code{ 4, -3, 2, 1 }, grad (4u, 0),
          loss_code (1u, 0), loss_grad (1u, 0), pool (pool_backing,
                                                     sizeof (pool_backing)),
          tape (tape_nodes, 4u, pool)
    {
        views.reserve (1u);
        views.push_back (Trainable::make (code.data (), grad.data (),
                                         Shape::vector (4u), Flag::RequiresGrad));
        ctx._M_pool = &pool;
        ctx._M_tape = &tape;
        ctx._M_params = views.data ();
        ctx._M_param_count = 1u;
        ctx._M_batch = 1u;
        ctx._M_loss = Trainable::make (loss_code.data (), loss_grad.data (),
                                      Shape::scalar (), 0u);
    }
};

[[nodiscard]] StepHooks
traj_hooks () noexcept
{
    StepHooks hooks;
    hooks._M_forward = traj_forward;
    hooks._M_update = traj_update;
    return hooks;
}

} // namespace

TEST_CASE ("copy_out exports a parameter into a model weight span")
{
    const ThreeParams fixture;
    std::vector<uint8_t> model (32u, 0x5Au);

    SECTION ("codes go in stride-1 and the unused tail is zeroed")
        {
            REQUIRE (status_ok (copy_out (fixture.views[0], model.data (), 32u, 4u,
                                          16u)));
            for (uint32_t i = 0u; i < 12u; ++i)
                REQUIRE (model[4u + i] == static_cast<uint8_t> (fixture.w_code[i]));
            for (uint32_t i = 12u; i < 16u; ++i)
                REQUIRE (model[4u + i] == 0u);
            REQUIRE (model[3u] == 0x5Au);
            REQUIRE (model[20u] == 0x5Au);
        }

    SECTION ("an exact-fit span writes every byte and zeroes none")
        {
            std::vector<uint8_t> tight (12u, 0x33u);
            REQUIRE (status_ok (copy_out (fixture.views[0], tight.data (), 12u, 0u,
                                          12u)));
            for (uint32_t i = 0u; i < 12u; ++i)
                REQUIRE (tight[i] == static_cast<uint8_t> (fixture.w_code[i]));
        }

    SECTION ("a span the layer declares smaller than the tensor is OUT_OF_RANGE")
        {
            REQUIRE (copy_out (fixture.views[0], model.data (), 32u, 0u, 11u)
                    == Status::OUT_OF_RANGE);
            REQUIRE (copy_out (fixture.views[1], model.data (), 32u, 0u, 2u)
                    == Status::OUT_OF_RANGE);
        }

    SECTION ("a span that leaves the model is OUT_OF_RANGE")
        {
            REQUIRE (copy_out (fixture.views[0], model.data (), 32u, 24u, 12u)
                    == Status::OUT_OF_RANGE);
            REQUIRE (copy_out (fixture.views[0], model.data (), 32u, 21u, 12u)
                    == Status::OUT_OF_RANGE);
            /* the offset arithmetic must not wrap */
            REQUIRE (copy_out (fixture.views[0], model.data (), 32u, 0xFFFFFFF0u,
                              12u)
                     == Status::OUT_OF_RANGE);
            REQUIRE (copy_out (fixture.views[0], model.data (), 0u, 0u, 12u)
                     == Status::OUT_OF_RANGE);
        }

    SECTION ("a -128 code is CORRUPT_STREAM, exported up to that byte")
        {
            ThreeParams local;
            local.w_code[5] = -128;
            std::vector<uint8_t> out (32u, 0x5Au);
            REQUIRE (copy_out (local.views[0], out.data (), 32u, 0u, 16u)
                    == Status::CORRUPT_STREAM);
            /* observed behaviour: the scan is streaming, so the five legal
               prefix bytes are already written when the rail aborts the export */
            for (uint32_t i = 0u; i < 5u; ++i)
                REQUIRE (out[i] == static_cast<uint8_t> (local.w_code[i]));
            REQUIRE (out[5] == 0x5Au);
            REQUIRE (out[6] == 0x5Au);
        }

    SECTION ("null, misaligned and invalid arguments are BAD_LAYOUT")
        {
            REQUIRE (copy_out (fixture.views[0], nullptr, 32u, 0u, 12u)
                    == Status::BAD_LAYOUT);
            REQUIRE (copy_out (fixture.views[0], model.data () + 1u, 31u, 0u, 12u)
                    == Status::BAD_LAYOUT);
            ThreeParams local;
            local.views[0]._M_count = 11u;
            REQUIRE (copy_out (local.views[0], model.data (), 32u, 0u, 16u)
                    == Status::BAD_LAYOUT);
        }
}

TEST_CASE ("copy_in imports a model weight stream back into a parameter")
{
    std::vector<uint8_t> model (24u, 0u);
    for (uint32_t i = 0u; i < 12u; ++i)
        model[i] = static_cast<uint8_t> (static_cast<int8_t> (i * 3 - 12));

    SECTION ("round trip through copy_out preserves every code")
        {
            ThreeParams source;
            REQUIRE (status_ok (copy_out (source.views[0], model.data (), 24u, 0u,
                                         12u)));
            ThreeParams sink;
            sink.scratch_codes (17);
            REQUIRE (status_ok (copy_in (model.data (), 24u, 0u, sink.views[0])));
            REQUIRE (sink.w_code == source.w_code);
            for (uint32_t i = 0u; i < 12u; ++i)
                REQUIRE (sink.views[0].code_at (i) == sink.w_code[i]);
        }

    SECTION ("the import only touches the tensor's own span")
        {
            ThreeParams sink;
            const int8_t keep = sink.w_code[0];
            REQUIRE (status_ok (copy_in (model.data (), 24u, 12u, sink.views[1])));
            REQUIRE (sink.b_code[0] == static_cast<int8_t> (model[12u]));
            REQUIRE (sink.w_code[0] == keep);
        }

    SECTION ("a -128 byte in the stream is CORRUPT_STREAM")
        {
            std::vector<uint8_t> bad = model;
            bad[3] = 0x80u;
            ThreeParams sink;
            sink.scratch_codes (9);
            REQUIRE (copy_in (bad.data (), 24u, 0u, sink.views[0])
                    == Status::CORRUPT_STREAM);
            /* observed behaviour: the import is streaming, so the legal prefix
               is already in place when the rail aborts it */
            REQUIRE (sink.w_code[0] == static_cast<int8_t> (bad[0]));
            REQUIRE (sink.w_code[1] == static_cast<int8_t> (bad[1]));
            REQUIRE (sink.w_code[2] == static_cast<int8_t> (bad[2]));
            REQUIRE (sink.w_code[3] == 9);
            REQUIRE (sink.w_code[4] == 9);
        }

    SECTION ("null, misaligned and out-of-range are rejected")
        {
            ThreeParams sink;
            REQUIRE (copy_in (nullptr, 24u, 0u, sink.views[0]) == Status::BAD_LAYOUT);
            REQUIRE (copy_in (model.data () + 1u, 23u, 0u, sink.views[0])
                    == Status::BAD_LAYOUT);
            REQUIRE (copy_in (model.data (), 24u, 20u, sink.views[0])
                    == Status::OUT_OF_RANGE);
            REQUIRE (copy_in (model.data (), 4u, 0u, sink.views[0])
                    == Status::OUT_OF_RANGE);
            sink.views[0]._M_code = nullptr;
            REQUIRE (copy_in (model.data (), 24u, 0u, sink.views[0])
                    == Status::BAD_LAYOUT);
        }
}

TEST_CASE ("a checkpoint restore resumes a training run byte-identically")
{
    /* Two independent trainers over the same toy run: one trains six steps, the
       other trains three, checkpoints, and is restored into a fresh trainer that
       trains the remaining three.  Both parameter buffers must agree exactly,
       and so must the step index, the loss and the loss scale. */
    TrajFixture long_run;
    TrajFixture save_side;
    TrajFixture resume_side;

    const StepHooks hooks = traj_hooks ();
    Trainer trainer_a (long_run.ctx, hooks);
    REQUIRE (status_ok (trainer_a.configure (long_run.views.data (), 1u, 1u)));
    trainer_a.set_loss_scale (16384);
    REQUIRE (trainer_a.loss_scale () == 16384);

    std::vector<std::vector<int8_t>> reference;
    reference.push_back (long_run.code);
    {
        std::vector<int8_t> codes = long_run.code;
        for (uint32_t step = 0u; step < 6u; ++step)
            {
                int32_t loss = 0;
                ref_traj_step (codes, step, loss);
                reference.push_back (codes);
            }
    }

    for (uint32_t step = 0u; step < 6u; ++step)
        {
            StepStats stats;
            REQUIRE (status_ok (trainer_a.step (stats)));
            REQUIRE (stats._M_step == step);
            REQUIRE (stats._M_loss_q7 == ref_traj_loss (reference[step].data (),
                                                        step));
            REQUIRE (stats._M_nodes == 1u);
            REQUIRE (stats._M_saturation_events == 0u);
            REQUIRE (stats._M_skipped_updates == 0u);
            REQUIRE (long_run.code == reference[step + 1u]);
        }
    const std::vector<int8_t> six_steps = long_run.code;

    SECTION ("a three-step checkpoint fed to a fresh trainer lands on the same bytes")
        {
            Trainer trainer_b (save_side.ctx, hooks);
            REQUIRE (status_ok (trainer_b.configure (save_side.views.data (), 1u,
                                                    1u)));
            trainer_b.set_loss_scale (16384);
            for (uint32_t step = 0u; step < 3u; ++step)
                {
                    StepStats stats;
                    REQUIRE (status_ok (trainer_b.step (stats)));
                }
            REQUIRE (save_side.code == reference[3u]);

            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (save_side.views.data (), 1u)));
            RunState saved;
            run_state_from_trainer (trainer_b, saved);
            uint32_t written = 0u;
            std::vector<uint8_t> blob (checkpoint_bytes (1u, 4u), 0u);
            REQUIRE (status_ok (ckpt.serialize (blob.data (),
                                                static_cast<uint32_t> (blob.size ()),
                                                saved, written)));

            /* the fresh side starts from a deliberately wrong state */
            Trainer trainer_c (resume_side.ctx, hooks);
            REQUIRE (status_ok (trainer_c.configure (resume_side.views.data (), 1u,
                                                    1u)));
            resume_side.code = {-100, 100, -7, 33};
            resume_side.ctx._M_step = 999u;
            trainer_c.set_loss_scale (2000);
            Checkpoint ckpt_c;
            REQUIRE (status_ok (ckpt_c.bind (resume_side.views.data (), 1u)));
            RunState restored;
            REQUIRE (status_ok (ckpt_c.restore (blob.data (), written, trainer_c,
                                                restored)));
            REQUIRE (restored._M_step == 3u);
            REQUIRE (restored._M_loss_scale_q15 == 16384);
            REQUIRE (resume_side.code == reference[3u]);
            REQUIRE (trainer_c.context ()._M_step == 3u);
            REQUIRE (trainer_c.loss_scale () == 16384);

            for (uint32_t step = 3u; step < 6u; ++step)
                {
                    StepStats stats;
                    REQUIRE (status_ok (trainer_c.step (stats)));
                    REQUIRE (stats._M_step == step);
                    REQUIRE (resume_side.code == reference[step + 1u]);
                }
            REQUIRE (resume_side.code == six_steps);
            REQUIRE (trainer_c.context ()._M_step == 6u);
            REQUIRE (trainer_c.loss_scale () == 16384);
        }

    SECTION ("run_state_from_trainer reports the trainer through its public view")
        {
            Trainer trainer (save_side.ctx, hooks);
            REQUIRE (status_ok (trainer.configure (save_side.views.data (), 1u,
                                                  1u)));
            RunState fresh;
            run_state_from_trainer (trainer, fresh);
            REQUIRE (fresh._M_step == 0u);
            REQUIRE (fresh._M_loss_scale_q15 == _S_q15_one);
            REQUIRE (fresh._M_clean_steps == 0u);
            REQUIRE (fresh._M_have_best == 0u);
            REQUIRE (fresh._M_best_loss_q7 == 0);
            REQUIRE (fresh._M_best_step == 0u);
            REQUIRE (fresh._M_epochs == 0u);
            REQUIRE (fresh._M_saturation_total == 0u);
            REQUIRE (fresh._M_scale_halves == 0u);
            REQUIRE (fresh._M_scale_grows == 0u);

            for (uint32_t step = 0u; step < 4u; ++step)
                {
                    StepStats stats;
                    REQUIRE (status_ok (trainer.step (stats)));
                }
            const FitStats totals = trainer.totals ();
            RunState mid;
            run_state_from_trainer (trainer, mid);
            REQUIRE (mid._M_step == totals._M_steps);
            REQUIRE (mid._M_step == 4u);
            REQUIRE (mid._M_first_loss_q7 == totals._M_first_loss_q7);
            REQUIRE (mid._M_last_loss_q7 == totals._M_last_loss_q7);
            REQUIRE (mid._M_best_loss_q7 == totals._M_best_loss_q7);
            REQUIRE (mid._M_best_step == totals._M_best_step);
            REQUIRE (mid._M_saturation_total == totals._M_saturation_events);
            REQUIRE (mid._M_have_best == 1u);
            /* _M_clean_steps has no public read path, so the serializer always
               writes zero, and totals () never fills _M_epochs either: both are
               pinned here because the doc calls them out. */
            REQUIRE (mid._M_clean_steps == 0u);
            REQUIRE (mid._M_epochs == 0u);
            int32_t best = ref_traj_loss (reference[0].data (), 0u);
            uint32_t best_step = 0u;
            for (uint32_t step = 0u; step < 4u; ++step)
                {
                    const int32_t loss = ref_traj_loss (reference[step].data (),
                                                        step);
                    if (loss < best)
                        {
                            best = loss;
                            best_step = step;
                        }
                }
            REQUIRE (mid._M_best_loss_q7 == best);
            REQUIRE (mid._M_best_step == best_step);
            REQUIRE (mid._M_last_loss_q7
                     == ref_traj_loss (reference[3u].data (), 3u));
        }

    SECTION ("a fit () run checkpoints its epoch and resumes its counters")
        {
            Trainer trainer (save_side.ctx, traj_hooks ());
            REQUIRE (status_ok (trainer.configure (save_side.views.data (), 1u,
                                                  1u)));
            trainer.set_loss_scale (16384);
            REQUIRE (trainer.loss_scale () == 16384);
            FitStats fit_stats;
            REQUIRE (status_ok (trainer.fit (3u, true, fit_stats)));
            REQUIRE (fit_stats._M_steps == 3u);
            REQUIRE (fit_stats._M_epochs == 1u);
            REQUIRE (fit_stats._M_status == Status::OK);
            REQUIRE (save_side.code == reference[3u]);
            RunState saved;
            run_state_from_trainer (trainer, saved);
            REQUIRE (saved._M_step == 3u);
            REQUIRE (saved._M_loss_scale_q15 == 16384);
            REQUIRE (saved._M_last_loss_q7 == fit_stats._M_last_loss_q7);

            Trainer next (resume_side.ctx, traj_hooks ());
            REQUIRE (status_ok (next.configure (resume_side.views.data (), 1u,
                                               1u)));
            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (resume_side.views.data (), 1u)));
            uint32_t written = 0u;
            std::vector<uint8_t> blob (checkpoint_bytes (1u, 4u), 0u);
            REQUIRE (status_ok (ckpt.serialize (blob.data (),
                                                static_cast<uint32_t> (blob.size ()),
                                                saved, written)));
            RunState restored;
            REQUIRE (status_ok (ckpt.restore (blob.data (), written, next,
                                             restored)));
            /* documented restore order: streams, then step/last loss, then the
               scale; the best/epoch counters are NOT pushed back in, so the
               resumed trainer rebuilds them from the continuation */
            REQUIRE (next.context ()._M_step == 3u);
            REQUIRE (next.context ()._M_loss_q7 == saved._M_last_loss_q7);
            REQUIRE (next.loss_scale () == saved._M_loss_scale_q15);
            REQUIRE (next.totals ()._M_best_loss_q7 == 0);
            REQUIRE (next.totals ()._M_first_loss_q7 == 0);
            /* the buffer keeps the scale word verbatim, but _S_q15_one is not a
               Q15 lane: set_loss_scale clamps it to _S_q15_max, so a run saved
               at the default scale resumes one unit lower. Pinned here because
               it is the one asymmetry in the round trip. */
            RunState unity;
            unity._M_loss_scale_q15 = _S_q15_one;
            run_state_to_trainer (next, unity);
            REQUIRE (next.loss_scale () == _S_q15_max);
            REQUIRE (unity._M_loss_scale_q15 == _S_q15_one);
            FitStats more;
            uint32_t steps_out = 0u;
            REQUIRE (status_ok (next.fit_until (3u, 40, true, more, steps_out)));
            REQUIRE (steps_out == 3u);
            REQUIRE (resume_side.code == six_steps);
        }

    SECTION ("a failed step after a restore still reports the restored geometry")
        {
            StepHooks broken = traj_hooks ();
            broken._M_forward = traj_forward_fails;
            Trainer trainer (save_side.ctx, broken);
            REQUIRE (status_ok (trainer.configure (save_side.views.data (), 1u,
                                                  1u)));
            StepStats stats;
            REQUIRE (trainer.step (stats) == Status::OUT_OF_RANGE);
            REQUIRE (stats._M_status == Status::OUT_OF_RANGE);
            REQUIRE (trainer.context ()._M_step == 0u);
            const std::vector<int8_t> untouched = save_side.code;
            Checkpoint ckpt;
            REQUIRE (status_ok (ckpt.bind (save_side.views.data (), 1u)));
            RunState saved;
            run_state_from_trainer (trainer, saved);
            REQUIRE (saved._M_have_best == 0u);
            uint32_t written = 0u;
            std::vector<uint8_t> blob (checkpoint_bytes (1u, 4u), 0u);
            REQUIRE (status_ok (ckpt.serialize (blob.data (),
                                                static_cast<uint32_t> (blob.size ()),
                                                saved, written)));
            Trainer good (save_side.ctx, traj_hooks ());
            REQUIRE (status_ok (good.configure (save_side.views.data (), 1u, 1u)));
            RunState restored;
            REQUIRE (status_ok (ckpt.restore (blob.data (), written, good,
                                             restored)));
            REQUIRE (save_side.code == untouched);
            REQUIRE (restored._M_have_best == 0u);
            StepStats ok_stats;
            REQUIRE (status_ok (good.step (ok_stats)));
            REQUIRE (save_side.code == reference[1u]);
        }

    SECTION ("run_state_to_trainer clamps the scale exactly like set_loss_scale")
        {
            Trainer trainer (save_side.ctx, traj_hooks ());
            REQUIRE (status_ok (trainer.configure (save_side.views.data (), 1u,
                                                  1u)));
            RunState state = sample_run ();
            state._M_loss_scale_q15 = _S_q15_one; /* 32768 is not a Q15 lane */
            run_state_to_trainer (trainer, state);
            REQUIRE (trainer.loss_scale () == _S_q15_max);
            REQUIRE (trainer.context ()._M_step == state._M_step);
            REQUIRE (trainer.context ()._M_loss_q7 == state._M_last_loss_q7);
            state._M_loss_scale_q15 = -300;
            run_state_to_trainer (trainer, state);
            REQUIRE (trainer.loss_scale () == -300);
            state._M_loss_scale_q15 = 0;
            run_state_to_trainer (trainer, state);
            REQUIRE (trainer.loss_scale () == 0);
        }
}

/* ==================================================================== *
 * Part B -- portable weight container ("MLCP")                          *
 * ==================================================================== */

namespace {

/** @brief A container tensor described the way the format documents it. */
struct RefWckTensor
{
    std::vector<int8_t> codes;
    std::vector<int16_t> grads;   /* empty = no lane stream in the blob */
    std::vector<int16_t> opt;     /* empty = no optimizer region        */
    std::vector<int32_t> scales;  /* empty = no multiplier table        */
    uint32_t scale_axis = _S_wck_per_tensor;
    uint16_t dims[4] = { 1u, 1u, 1u, 1u };
    uint8_t rank = 1u;
    uint16_t flags = 0u;
    int32_t scale_q15 = _S_q15_one;
};

struct RefWckGeom
{
    uint32_t table = 0u;
    uint32_t scale = 0u;
    uint32_t code = 0u;
    uint32_t grad = 0u;
    uint32_t opt = 0u;
    uint32_t payload = 0u;
    uint32_t total = 0u;
    uint32_t content = 0u;
    uint32_t padding = 0u;
    uint32_t scale_entries = 0u;
    uint8_t flags = 0u;
};

[[nodiscard]] RefWckGeom
ref_wck_geom (const std::vector<RefWckTensor> &tensors, const bool is_best) noexcept
{
    RefWckGeom g;
    g.table = static_cast<uint32_t> (tensors.size ()) * _S_wck_entry_bytes;
    for (const RefWckTensor &t : tensors)
        {
            g.code += ref_align4 (static_cast<uint32_t> (t.codes.size ()));
            g.content += static_cast<uint32_t> (t.codes.size ());
            if (!t.grads.empty ())
                {
                    g.grad += ref_align4 (static_cast<uint32_t> (t.grads.size ()) * 2u);
                    g.content += static_cast<uint32_t> (t.grads.size ()) * 2u;
                    g.flags |= static_cast<uint8_t> (WckFlag::WckHasGrad);
                }
            if (!t.opt.empty ())
                {
                    g.opt += ref_align4 (static_cast<uint32_t> (t.opt.size ()) * 2u);
                    g.content += static_cast<uint32_t> (t.opt.size ()) * 2u;
                    g.flags |= static_cast<uint8_t> (WckFlag::WckHasOpt);
                }
            if (!t.scales.empty ())
                {
                    g.scale += static_cast<uint32_t> (t.scales.size ())
                               * _S_wck_scale_entry_bytes;
                    g.content += static_cast<uint32_t> (t.scales.size ())
                                 * _S_wck_scale_entry_bytes;
                    g.scale_entries += static_cast<uint32_t> (t.scales.size ());
                    g.flags |= static_cast<uint8_t> (WckFlag::WckHasScaleTable);
                }
        }
    if (is_best)
        g.flags |= static_cast<uint8_t> (WckFlag::WckIsBest);
    g.payload = _S_wck_reserved_bytes + g.scale + g.code + g.grad + g.opt;
    g.total = _S_wck_header_bytes + g.table + g.payload;
    g.padding = g.payload - g.content;
    return g;
}

/** @brief A completely independent container writer: same rules, own bytes. */
[[nodiscard]] std::vector<uint8_t>
ref_wck_build (const std::vector<RefWckTensor> &tensors, const uint32_t step,
               const int32_t loss_scale_q15, const bool is_best) noexcept
{
    const RefWckGeom g = ref_wck_geom (tensors, is_best);
    std::vector<uint8_t> blob (g.total, 0u);
    uint8_t *const head = blob.data ();
    head[0] = static_cast<uint8_t> ('M');
    head[1] = static_cast<uint8_t> ('L');
    head[2] = static_cast<uint8_t> ('C');
    head[3] = static_cast<uint8_t> ('P');
    head[4] = _S_wck_version;
    head[5] = g.flags;
    wr16 (head + 6, static_cast<uint16_t> (tensors.size ()));
    wr32 (head + 8, g.table);
    wr32 (head + 12, g.scale);
    wr32 (head + 16, g.code);
    wr32 (head + 20, g.grad);
    wr32 (head + 24, g.opt);
    wr32 (head + 28, g.payload);
    wr32 (head + 32, g.total);
    wr32 (head + 36, step);
    wr32 (head + 40, static_cast<uint32_t> (loss_scale_q15));
    wr16 (head + 44, _S_wck_header_bytes);
    wr16 (head + 50, static_cast<uint16_t> (g.scale_entries));

    uint32_t scale_run = _S_wck_reserved_bytes;
    uint32_t code_run = _S_wck_reserved_bytes + g.scale;
    uint32_t grad_run = code_run + g.code;
    uint32_t opt_run = grad_run + g.grad;
    const uint32_t payload_off = _S_wck_header_bytes + g.table;
    for (uint32_t index = 0u; index < tensors.size (); ++index)
        {
            const RefWckTensor &t = tensors[index];
            uint8_t *const at = blob.data () + _S_wck_header_bytes
                                + index * _S_wck_entry_bytes;
            const uint32_t count = static_cast<uint32_t> (t.codes.size ());
            const uint32_t code_off = code_run;
            code_run += ref_align4 (count);
            const uint32_t grad_off = t.grads.empty () ? 0u : grad_run;
            if (!t.grads.empty ())
                grad_run += ref_align4 (static_cast<uint32_t> (t.grads.size ()) * 2u);
            const uint32_t opt_off = t.opt.empty () ? 0u : opt_run;
            if (!t.opt.empty ())
                opt_run += ref_align4 (static_cast<uint32_t> (t.opt.size ()) * 2u);
            const uint32_t scale_off = t.scales.empty () ? 0u : scale_run;
            if (!t.scales.empty ())
                scale_run += static_cast<uint32_t> (t.scales.size ())
                             * _S_wck_scale_entry_bytes;

            for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
                wr16 (at + axis * 2u, t.dims[axis]);
            at[8] = t.rank;
            wr16 (at + 10, t.flags);
            wr32 (at + 12, count);
            wr32 (at + 16, static_cast<uint32_t> (t.scale_q15));
            wr32 (at + 20, code_off);
            wr32 (at + 24, grad_off);
            wr32 (at + 28, opt_off);
            wr32 (at + 32, scale_off);
            wr16 (at + 36, static_cast<uint16_t> (t.scales.size ()));
            wr32 (at + 40, static_cast<uint32_t> (t.opt.size ()));

            for (uint32_t i = 0u; i < count; ++i)
                blob[payload_off + code_off + i]
                    = static_cast<uint8_t> (t.codes[i]);
            for (uint32_t i = 0u; i < t.grads.size (); ++i)
                wr16 (blob.data () + payload_off + grad_off + 2u * i,
                      static_cast<uint16_t> (t.grads[i]));
            for (uint32_t i = 0u; i < t.opt.size (); ++i)
                wr16 (blob.data () + payload_off + opt_off + 2u * i,
                      static_cast<uint16_t> (t.opt[i]));
            for (uint32_t i = 0u; i < t.scales.size (); ++i)
                wr32 (blob.data () + payload_off + scale_off + 4u * i,
                      static_cast<uint32_t> (t.scales[i]));
        }

    wr16 (head + 46, ref_crc16 (blob.data () + payload_off, g.payload));
    wr16 (head + 48, ref_crc16 (head, 48u));
    return blob;
}

/** @brief Two tensors: one fully instrumented, one bare bias-like vector. */
struct WckFixture
{
    std::vector<int8_t> a_code;
    std::vector<int8_t> b_code;
    std::vector<int16_t> a_grad;
    std::vector<int16_t> a_opt;
    std::vector<int32_t> a_scale;
    std::vector<int32_t> b_scale;
    std::vector<Trainable> views;
    std::vector<WckTensorSource> sources;
    uint32_t step = 91u;
    int32_t loss_scale_q15 = 16384;

    WckFixture ()
        : a_code (5u, 0), b_code (3u, 0), a_grad (5u, 0), a_opt (5u, 0),
          a_scale (1u, 0), b_scale (3u, 0)
    {
        for (uint32_t i = 0u; i < 5u; ++i)
            {
                a_code[i] = static_cast<int8_t> (i * 11 - 20);
                a_grad[i] = static_cast<int16_t> (1024u * (i + 1u));
                a_opt[i] = static_cast<int16_t> (-512 - static_cast<int32_t> (i) * 3);
            }
        for (uint32_t i = 0u; i < 3u; ++i)
            b_code[i] = static_cast<int8_t> (-7 + static_cast<int32_t> (i) * 5);
        a_scale[0] = 16384;
        for (uint32_t i = 0u; i < 3u; ++i)
            b_scale[i] = static_cast<int32_t> (2000 + i * 700);
        views.reserve (2u);
        views.push_back (Trainable::make (
            a_code.data (), a_grad.data (), Shape::vector (5u),
            Flag::RequiresGrad | Flag::Quantized));
        views.push_back (Trainable::make (
            b_code.data (), nullptr, Shape::vector (3u),
            Flag::Frozen | Flag::BiasLike));
        views[0]._M_scale_q15 = 16000;
        views[1]._M_scale_q15 = 24000;
        sources.resize (2u);
        sources[0]._M_param = &views[0];
        sources[0]._M_opt = a_opt.data ();
        sources[0]._M_opt_lanes = 5u;
        sources[0]._M_scale_table = a_scale.data ();
        sources[0]._M_scale_entries = 1u;
        sources[0]._M_scale_axis = _S_wck_per_tensor;
        sources[1]._M_param = &views[1];
        sources[1]._M_scale_table = b_scale.data ();
        sources[1]._M_scale_entries = 3u;
        sources[1]._M_scale_axis = 0u;
    }

    /** @brief The same pair expressed for the reference writer. */
    [[nodiscard]] std::vector<RefWckTensor>
    describe () const noexcept
    {
        std::vector<RefWckTensor> out (2u);
        out[0].codes = a_code;
        out[0].grads = a_grad;
        out[0].opt = a_opt;
        out[0].scales = a_scale;
        out[0].dims[0] = 5u;
        out[0].rank = 1u;
        out[0].flags = static_cast<uint16_t> (views[0]._M_flags);
        out[0].scale_q15 = views[0]._M_scale_q15;
        out[0].scale_axis = _S_wck_per_tensor;
        out[1].codes = b_code;
        out[1].scales = b_scale;
        out[1].dims[0] = 3u;
        out[1].rank = 1u;
        out[1].flags = static_cast<uint16_t> (views[1]._M_flags);
        out[1].scale_q15 = views[1]._M_scale_q15;
        out[1].scale_axis = 0u;
        return out;
    }
};

[[nodiscard]] std::vector<uint8_t>
write_wck (const WckFixture &fixture, WckWriteStats &stats,
           const bool is_best = false) noexcept
{
    const RefWckGeom g = ref_wck_geom (fixture.describe (), is_best);
    std::vector<uint8_t> blob (g.total, 0x5Au);
    const Status status = wck_write (blob.data (), g.total, fixture.sources.data (),
                                    static_cast<uint32_t> (fixture.sources.size ()),
                                    fixture.step, fixture.loss_scale_q15, stats,
                                    is_best);
    REQUIRE (status_ok (status));
    return blob;
}

} // namespace

TEST_CASE ("weight container wire constants and POD sizes")
{
    static_assert (sizeof (WckHeader) == 56u, "header wire size");
    static_assert (sizeof (WckEntry) == 48u, "entry wire size");
    static_assert (sizeof (WckWriteStats) == 24u, "stats are six words");
    static_assert (sizeof (WckTensorSource)
                       == (sizeof (void *) == 4u ? 28u : 40u),
                   "source size");
    static_assert (sizeof (WckView)
                       == sizeof (WckHeader) + (sizeof (void *) == 4u ? 16u : 24u),
                   "view is the decoded header plus two offsets");

    SECTION ("magic is the four ASCII bytes MLCP, not a word")
        {
            REQUIRE (_S_wck_magic[0] == 'M');
            REQUIRE (_S_wck_magic[1] == 'L');
            REQUIRE (_S_wck_magic[2] == 'C');
            REQUIRE (_S_wck_magic[3] == 'P');
            REQUIRE (sizeof (_S_wck_magic) == 4u);
            REQUIRE (_S_wck_version == 1u);
        }

    SECTION ("sizes, limits and the per-tensor marker")
        {
            REQUIRE (_S_wck_header_bytes == 56u);
            REQUIRE (_S_wck_entry_bytes == 48u);
            REQUIRE (_S_wck_max_params == 1024u);
            REQUIRE (_S_wck_scale_entry_bytes == 4u);
            REQUIRE (_S_wck_reserved_bytes == 4u);
            REQUIRE (_S_wck_alignment == 4u);
            REQUIRE (_S_wck_max_payload == 8u * 1024u * 1024u);
            REQUIRE (_S_wck_max_scale_entries == 0xFFFFu);
            REQUIRE (_S_wck_per_tensor == 0xFFFFFFFFu);
        }

    SECTION ("flag bits are distinct single bits")
        {
            REQUIRE (static_cast<uint8_t> (WckFlag::WckHasGrad) == 1u);
            REQUIRE (static_cast<uint8_t> (WckFlag::WckHasOpt) == 2u);
            REQUIRE (static_cast<uint8_t> (WckFlag::WckHasScaleTable) == 4u);
            REQUIRE (static_cast<uint8_t> (WckFlag::WckIsBest) == 8u);
            WckHeader header;
            header._M_flags = static_cast<uint8_t> (WckFlag::WckHasGrad
                                                   | WckFlag::WckIsBest);
            REQUIRE (wck_has (header, WckFlag::WckHasGrad));
            REQUIRE (not wck_has (header, WckFlag::WckHasOpt));
            REQUIRE (not wck_has (header, WckFlag::WckHasScaleTable));
            REQUIRE (wck_has (header, WckFlag::WckIsBest));
        }

    SECTION ("the reference field order is the wire order")
        {
            WckHeader header;
            REQUIRE (offsetof (WckHeader, _M_magic) == 0u);
            REQUIRE (offsetof (WckHeader, _M_version) == 4u);
            REQUIRE (offsetof (WckHeader, _M_flags) == 5u);
            REQUIRE (offsetof (WckHeader, _M_param_count) == 6u);
            REQUIRE (offsetof (WckHeader, _M_table_bytes) == 8u);
            REQUIRE (offsetof (WckHeader, _M_scale_bytes) == 12u);
            REQUIRE (offsetof (WckHeader, _M_code_bytes) == 16u);
            REQUIRE (offsetof (WckHeader, _M_grad_bytes) == 20u);
            REQUIRE (offsetof (WckHeader, _M_opt_bytes) == 24u);
            REQUIRE (offsetof (WckHeader, _M_payload_bytes) == 28u);
            REQUIRE (offsetof (WckHeader, _M_total_bytes) == 32u);
            REQUIRE (offsetof (WckHeader, _M_step) == 36u);
            REQUIRE (offsetof (WckHeader, _M_loss_scale_q15) == 40u);
            REQUIRE (offsetof (WckHeader, _M_header_size) == 44u);
            REQUIRE (offsetof (WckHeader, _M_payload_crc) == 46u);
            REQUIRE (offsetof (WckHeader, _M_header_crc) == 48u);
            REQUIRE (offsetof (WckHeader, _M_scale_entry_count) == 50u);
            REQUIRE (offsetof (WckHeader, _M_reserved) == 52u);
            REQUIRE (offsetof (WckEntry, _M_dims) == 0u);
            REQUIRE (offsetof (WckEntry, _M_rank) == 8u);
            REQUIRE (offsetof (WckEntry, _M_flags) == 10u);
            REQUIRE (offsetof (WckEntry, _M_count) == 12u);
            REQUIRE (offsetof (WckEntry, _M_scale_q15) == 16u);
            REQUIRE (offsetof (WckEntry, _M_code_off) == 20u);
            REQUIRE (offsetof (WckEntry, _M_grad_off) == 24u);
            REQUIRE (offsetof (WckEntry, _M_opt_off) == 28u);
            REQUIRE (offsetof (WckEntry, _M_scale_off) == 32u);
            REQUIRE (offsetof (WckEntry, _M_scale_entries) == 36u);
            REQUIRE (offsetof (WckEntry, _M_opt_lanes) == 40u);
            REQUIRE (offsetof (WckEntry, _M_reserved) == 44u);
            REQUIRE (header._M_loss_scale_q15 == _S_q15_one);
        }

    SECTION ("the region helpers implement the documented alignment")
        {
            REQUIRE (wck_align_span (0u) == 0u);
            REQUIRE (wck_align_span (1u) == 4u);
            REQUIRE (wck_align_span (4u) == 4u);
            REQUIRE (wck_align_span (5u) == 8u);
            REQUIRE (wck_align_span (10u) == 12u);
            REQUIRE (wck_aligned4 (0u));
            REQUIRE (wck_aligned4 (4u));
            REQUIRE (not wck_aligned4 (1u));
            REQUIRE (not wck_aligned4 (6u));
        }
}

TEST_CASE ("wck_write accounts every region and matches an independent writer")
{
    const WckFixture fixture;
    const std::vector<RefWckTensor> desc = fixture.describe ();
    const RefWckGeom geometry = ref_wck_geom (desc, false);
    WckWriteStats stats;
    const std::vector<uint8_t> blob = write_wck (fixture, stats);

    SECTION ("the reference geometry is the documented 208-byte container")
        {
            REQUIRE (geometry.table == 96u);
            REQUIRE (geometry.scale == 16u);
            REQUIRE (geometry.code == 12u);
            REQUIRE (geometry.grad == 12u);
            REQUIRE (geometry.opt == 12u);
            REQUIRE (geometry.payload == 56u);
            REQUIRE (geometry.total == 208u);
            REQUIRE (geometry.content == 44u);
            REQUIRE (geometry.padding == 12u);
            REQUIRE (geometry.scale_entries == 4u);
            REQUIRE (geometry.flags
                     == static_cast<uint8_t> (WckFlag::WckHasGrad
                                              | WckFlag::WckHasOpt
                                              | WckFlag::WckHasScaleTable));
        }

    SECTION ("the write statistics agree with the reference accounting")
        {
            REQUIRE (stats._M_bytes_written == geometry.total);
            REQUIRE (stats._M_code_bytes == geometry.code);
            REQUIRE (stats._M_grad_bytes == geometry.grad);
            REQUIRE (stats._M_opt_bytes == geometry.opt);
            REQUIRE (stats._M_scale_bytes == geometry.scale);
            REQUIRE (stats._M_padding_bytes == geometry.padding);
            /* padding is the reserved word plus every alignment slack: the
               streams carry 8+10+10+16 = 44 data bytes in 56 payload bytes */
            REQUIRE (stats._M_padding_bytes
                     == _S_wck_reserved_bytes + (8u - 5u) + (4u - 3u)
                        + (12u - 10u) + (12u - 10u));
        }

    SECTION ("a blob written by the library is byte-identical to one written by hand")
        {
            const std::vector<uint8_t> mine
                = ref_wck_build (desc, fixture.step, fixture.loss_scale_q15, false);
            REQUIRE (mine.size () == blob.size ());
            for (uint32_t i = 0u; i < blob.size (); ++i)
                REQUIRE (blob[i] == mine[i]);
        }

    SECTION ("the header words carry the run, not only the geometry")
        {
            REQUIRE (rd16 (blob.data () + 6) == 2u);
            REQUIRE (rd32 (blob.data () + 36) == fixture.step);
            REQUIRE (static_cast<int32_t> (rd32 (blob.data () + 40))
                     == fixture.loss_scale_q15);
            REQUIRE (rd16 (blob.data () + 44) == _S_wck_header_bytes);
            REQUIRE (rd16 (blob.data () + 50) == 4u);
            REQUIRE (rd32 (blob.data () + 52) == 0u);
            REQUIRE (rd16 (blob.data () + 46)
                     == ref_crc16 (blob.data () + 152u, geometry.payload));
            REQUIRE (rd16 (blob.data () + 48) == ref_crc16 (blob.data (), 48u));
        }

    SECTION ("slack bytes past the container are left alone")
        {
            std::vector<uint8_t> roomy (geometry.total + 64u, 0x5Au);
            WckWriteStats local;
            REQUIRE (status_ok (wck_write (roomy.data (),
                                          static_cast<uint32_t> (roomy.size ()),
                                          fixture.sources.data (), 2u,
                                          fixture.step,
                                          fixture.loss_scale_q15, local)));
            REQUIRE (local._M_bytes_written == geometry.total);
            REQUIRE (roomy[geometry.total] == 0x5Au);
            REQUIRE (roomy[roomy.size () - 1u] == 0x5Au);
            for (uint32_t i = 0u; i < geometry.total; ++i)
                REQUIRE (roomy[i] == blob[i]);
        }

    SECTION ("a capacity below the need is TRUNCATED and writes nothing")
        {
            std::vector<uint8_t> short_buf (geometry.total - 1u, 0x5Au);
            WckWriteStats local;
            REQUIRE (wck_write (short_buf.data (),
                                static_cast<uint32_t> (short_buf.size ()),
                                fixture.sources.data (), 2u, fixture.step,
                                fixture.loss_scale_q15, local)
                     == Status::TRUNCATED);
            REQUIRE (local._M_bytes_written == 0u);
            REQUIRE (local._M_padding_bytes == 0u);
            for (uint8_t byte : short_buf)
                REQUIRE (byte == 0x5Au);
        }

    SECTION ("null and misaligned destinations are BAD_LAYOUT")
        {
            WckWriteStats local;
            REQUIRE (wck_write (nullptr, geometry.total, fixture.sources.data (),
                               2u, fixture.step, fixture.loss_scale_q15, local)
                     == Status::BAD_LAYOUT);
            std::vector<uint8_t> buf (geometry.total, 0u);
            REQUIRE (wck_write (buf.data () + 1u, geometry.total - 1u,
                                fixture.sources.data (), 2u, fixture.step,
                                fixture.loss_scale_q15, local)
                     == Status::BAD_LAYOUT);
            REQUIRE (local._M_bytes_written == 0u);
        }

    SECTION ("the best-loss flag is the only difference an is_best write makes")
        {
            WckWriteStats best_stats;
            const std::vector<uint8_t> best = write_wck (fixture, best_stats, true);
            REQUIRE (best.size () == blob.size ());
            uint32_t differing = 0u;
            for (uint32_t i = 0u; i < best.size (); ++i)
                if (best[i] != blob[i])
                    ++differing;
            /* byte 5 (flags) and the two CRCs resealed over it */
            REQUIRE (differing == 3u);
            REQUIRE (rd16 (best.data () + 6) == 2u);
            REQUIRE (best[5]
                     == static_cast<uint8_t> (
                            geometry.flags | static_cast<uint8_t> (WckFlag::WckIsBest)));
            WckView view;
            REQUIRE (status_ok (wck_read (best.data (),
                                          static_cast<uint32_t> (best.size ()),
                                          view)));
            REQUIRE (wck_has (wck_header (view), WckFlag::WckIsBest));
            REQUIRE (view._M_header._M_step == fixture.step);
        }
}

TEST_CASE ("wck_read accepts a container it did not write")
{
    const WckFixture fixture;
    const std::vector<RefWckTensor> desc = fixture.describe ();
    const std::vector<uint8_t> blob
        = ref_wck_build (desc, 1234u, 8192, true); /* written offline, sealed by hand */
    WckView view;
    const Status status
        = wck_read (blob.data (), static_cast<uint32_t> (blob.size ()), view);

    SECTION ("the view is valid and describes the blob")
        {
            REQUIRE (status_ok (status));
            REQUIRE (view._M_valid == 1u);
            REQUIRE (view._M_data == blob.data ());
            REQUIRE (view._M_length == blob.size ());
            REQUIRE (view._M_payload_off == 152u);
            const WckHeader &header = wck_header (view);
            REQUIRE (header._M_version == _S_wck_version);
            REQUIRE (header._M_param_count == 2u);
            REQUIRE (header._M_table_bytes == 96u);
            REQUIRE (header._M_scale_bytes == 16u);
            REQUIRE (header._M_code_bytes == 12u);
            REQUIRE (header._M_grad_bytes == 12u);
            REQUIRE (header._M_opt_bytes == 12u);
            REQUIRE (header._M_payload_bytes == 56u);
            REQUIRE (header._M_total_bytes == 208u);
            REQUIRE (header._M_step == 1234u);
            REQUIRE (header._M_loss_scale_q15 == 8192);
            REQUIRE (header._M_scale_entry_count == 4u);
            REQUIRE (header._M_reserved == 0u);
            REQUIRE (header._M_header_size == _S_wck_header_bytes);
            REQUIRE (wck_has (header, WckFlag::WckHasGrad));
            REQUIRE (wck_has (header, WckFlag::WckHasOpt));
            REQUIRE (wck_has (header, WckFlag::WckHasScaleTable));
            REQUIRE (wck_has (header, WckFlag::WckIsBest));
        }

    SECTION ("entries decode shape, count, scale and every region offset")
        {
            REQUIRE (status_ok (status));
            WckEntry entry;
            REQUIRE (status_ok (wck_entry (view, 0u, entry)));
            REQUIRE (entry._M_dims[0] == 5u);
            REQUIRE (entry._M_dims[1] == 1u);
            REQUIRE (entry._M_rank == 1u);
            REQUIRE (entry._M_count == 5u);
            REQUIRE (entry._M_scale_q15 == 16000);
            REQUIRE (entry._M_flags == static_cast<uint16_t> (fixture.views[0]._M_flags));
            REQUIRE (entry._M_code_off == 20u);
            REQUIRE (entry._M_grad_off == 32u);
            REQUIRE (entry._M_opt_off == 44u);
            REQUIRE (entry._M_scale_off == 4u);
            REQUIRE (entry._M_scale_entries == 1u);
            REQUIRE (entry._M_opt_lanes == 5u);
            REQUIRE (entry._M_pad == 0u);
            REQUIRE (entry._M_pad2 == 0u);
            REQUIRE (entry._M_reserved == 0u);

            REQUIRE (status_ok (wck_entry (view, 1u, entry)));
            REQUIRE (entry._M_dims[0] == 3u);
            REQUIRE (entry._M_count == 3u);
            REQUIRE (entry._M_scale_q15 == 24000);
            REQUIRE (entry._M_code_off == 28u);
            REQUIRE (entry._M_grad_off == 0u);
            REQUIRE (entry._M_opt_off == 0u);
            REQUIRE (entry._M_scale_off == 8u);
            REQUIRE (entry._M_scale_entries == 3u);
            REQUIRE (entry._M_opt_lanes == 0u);

            REQUIRE (wck_entry (view, 2u, entry) == Status::OUT_OF_RANGE);
            REQUIRE (entry._M_count == 0u); /* cleared on rejection */
        }

    SECTION ("the region accessors alias the payload, not a copy")
        {
            REQUIRE (status_ok (status));
            const uint8_t *codes = nullptr;
            uint32_t length = 0u;
            REQUIRE (status_ok (wck_code_span (view, 0u, codes, length)));
            REQUIRE (length == 5u);
            REQUIRE (codes == blob.data () + 152u + 20u);
            for (uint32_t i = 0u; i < 5u; ++i)
                REQUIRE (codes[i] == static_cast<uint8_t> (fixture.a_code[i]));
            REQUIRE (status_ok (wck_code_span (view, 1u, codes, length)));
            REQUIRE (length == 3u);
            REQUIRE (codes == blob.data () + 152u + 28u);

            const int16_t *grads = nullptr;
            uint32_t lanes = 0u;
            REQUIRE (status_ok (wck_grad_span (view, 0u, grads, lanes)));
            REQUIRE (lanes == 5u);
            for (uint32_t i = 0u; i < 5u; ++i)
                REQUIRE (grads[i] == fixture.a_grad[i]);
            REQUIRE (wck_grad_span (view, 1u, grads, lanes) == Status::UNSUPPORTED);
            REQUIRE (grads == nullptr);
            REQUIRE (lanes == 0u);

            const int16_t *opt = nullptr;
            REQUIRE (status_ok (wck_opt_span (view, 0u, opt, lanes)));
            REQUIRE (lanes == 5u);
            for (uint32_t i = 0u; i < 5u; ++i)
                REQUIRE (opt[i] == fixture.a_opt[i]);
            REQUIRE (wck_opt_span (view, 1u, opt, lanes) == Status::UNSUPPORTED);

            REQUIRE (wck_code_span (view, 9u, codes, length)
                     == Status::OUT_OF_RANGE);
        }

    SECTION ("multiplier tables read per channel")
        {
            REQUIRE (status_ok (status));
            int32_t value = 0;
            REQUIRE (status_ok (wck_scale_at (view, 0u, 0u, value)));
            REQUIRE (value == 16384);
            for (uint32_t channel = 0u; channel < 3u; ++channel)
                {
                    REQUIRE (status_ok (wck_scale_at (view, 1u, channel, value)));
                    REQUIRE (value == fixture.b_scale[channel]);
                }
            REQUIRE (wck_scale_at (view, 1u, 3u, value) == Status::OUT_OF_RANGE);
            REQUIRE (value == 0);
            REQUIRE (wck_scale_at (view, 0u, 1u, value) == Status::OUT_OF_RANGE);
            REQUIRE (wck_scale_at (view, 2u, 0u, value) == Status::OUT_OF_RANGE);
            REQUIRE (wck_scale_at (view, 0u, 0xFFFFFFFFu, value)
                     == Status::OUT_OF_RANGE);
        }

    SECTION ("a longer buffer is fine, a shorter one is TRUNCATED")
        {
            std::vector<uint8_t> roomy = blob;
            roomy.resize (blob.size () + 37u, 0u);
            WckView padded;
            REQUIRE (status_ok (wck_read (roomy.data (),
                                         static_cast<uint32_t> (roomy.size ()),
                                         padded)));
            REQUIRE (padded._M_length == roomy.size ());
            REQUIRE (padded._M_header._M_total_bytes == 208u);

            WckView tight;
            REQUIRE (wck_read (blob.data (), 207u, tight) == Status::TRUNCATED);
            REQUIRE (tight._M_valid == 0u);
            REQUIRE (tight._M_data == nullptr);
            REQUIRE (wck_read (blob.data (), 55u, tight) == Status::TRUNCATED);
        }
}

TEST_CASE ("a hand-built minimal container round trips through the accessors")
{
    /* One tensor, four codes, a per-tensor multiplier, no gradients: the
       smallest legal container, assembled field by field with the reference
       byte writers. */
    std::vector<uint8_t> blob (116u, 0u);
    REQUIRE (blob.size ()
             == _S_wck_header_bytes + _S_wck_entry_bytes
                + _S_wck_reserved_bytes + _S_wck_scale_entry_bytes + 4u);
    blob[0] = static_cast<uint8_t> ('M');
    blob[1] = static_cast<uint8_t> ('L');
    blob[2] = static_cast<uint8_t> ('C');
    blob[3] = static_cast<uint8_t> ('P');
    blob[4] = _S_wck_version;
    blob[5] = static_cast<uint8_t> (WckFlag::WckHasScaleTable);
    wr16 (blob.data () + 6, 1u);            /* one parameter */
    wr32 (blob.data () + 8, 48u);           /* table bytes */
    wr32 (blob.data () + 12, 4u);           /* scale bytes */
    wr32 (blob.data () + 16, 4u);           /* code bytes */
    wr32 (blob.data () + 20, 0u);           /* no gradients */
    wr32 (blob.data () + 24, 0u);           /* no optimizer state */
    wr32 (blob.data () + 28, 12u);          /* payload: 4 + 4 + 4 */
    wr32 (blob.data () + 32, 116u);         /* total */
    wr32 (blob.data () + 36, 7u);           /* step */
    wr32 (blob.data () + 40, 4096u);        /* loss scale */
    wr16 (blob.data () + 44, 56u);          /* header size */
    wr16 (blob.data () + 50, 1u);           /* one multiplier */
    const uint32_t payload_off = _S_wck_header_bytes + _S_wck_entry_bytes;
    wr32 (blob.data () + payload_off + 4u, -20000); /* the table entry */
    /* entry: dims, rank, flags, count, scale, offsets */
    wr16 (blob.data () + 56u + 0u, 4u);
    wr16 (blob.data () + 56u + 2u, 1u);
    wr16 (blob.data () + 56u + 4u, 1u);
    wr16 (blob.data () + 56u + 6u, 1u);
    blob[56u + 8u] = 1u;                    /* rank */
    wr16 (blob.data () + 56u + 10u, static_cast<uint16_t> (Flag::Quantized));
    wr32 (blob.data () + 56u + 12u, 4u);    /* count */
    wr32 (blob.data () + 56u + 16u, static_cast<uint32_t> (12345));
    wr32 (blob.data () + 56u + 20u, 8u);    /* code offset */
    wr32 (blob.data () + 56u + 24u, 0u);    /* no grad stream */
    wr32 (blob.data () + 56u + 28u, 0u);    /* no opt stream */
    wr32 (blob.data () + 56u + 32u, 4u);    /* scale offset */
    wr16 (blob.data () + 56u + 36u, 1u);    /* one entry */
    wr32 (blob.data () + 56u + 40u, 0u);    /* no optimizer lanes */
    blob[payload_off + 8u] = static_cast<uint8_t> (static_cast<int8_t> (-3));
    blob[payload_off + 9u] = static_cast<uint8_t> (static_cast<int8_t> (0));
    blob[payload_off + 10u] = static_cast<uint8_t> (static_cast<int8_t> (7));
    blob[payload_off + 11u] = static_cast<uint8_t> (static_cast<int8_t> (127));
    reseal_wck (blob);

    WckView view;
    SECTION ("it validates and reports the geometry the bytes claim")
        {
            REQUIRE (status_ok (wck_read (blob.data (), 116u, view)));
            REQUIRE (view._M_payload_off == payload_off);
            REQUIRE (wck_header (view)._M_total_bytes == 116u);
            REQUIRE (not wck_has (wck_header (view), WckFlag::WckHasGrad));
            REQUIRE (wck_has (wck_header (view), WckFlag::WckHasScaleTable));
            const uint8_t *codes = nullptr;
            uint32_t length = 0u;
            REQUIRE (status_ok (wck_code_span (view, 0u, codes, length)));
            REQUIRE (length == 4u);
            REQUIRE (static_cast<int8_t> (codes[0]) == -3);
            REQUIRE (static_cast<int8_t> (codes[3]) == 127);
            int32_t scale = 0;
            REQUIRE (status_ok (wck_scale_at (view, 0u, 0u, scale)));
            REQUIRE (scale == -20000);
            const int16_t *lanes = nullptr;
            uint32_t count = 0u;
            REQUIRE (wck_grad_span (view, 0u, lanes, count) == Status::UNSUPPORTED);
            REQUIRE (wck_opt_span (view, 0u, lanes, count) == Status::UNSUPPORTED);
        }

    SECTION ("restore lands the codes, the scale and the flags on a live view")
        {
            REQUIRE (status_ok (wck_read (blob.data (), 116u, view)));
            std::vector<int8_t> code (4u, 0);
            std::vector<int16_t> grad (4u, 0);
            Trainable destination = Trainable::make (code.data (), grad.data (),
                                                    Shape::vector (4u),
                                                    Flag::RequiresGrad);
            REQUIRE (status_ok (wck_restore (view, 0u, destination)));
            REQUIRE (code[0] == -3);
            REQUIRE (code[3] == 127);
            REQUIRE (destination._M_scale_q15 == 12345);
            REQUIRE (destination._M_flags
                     == static_cast<uint32_t> (Flag::Quantized));
            REQUIRE (destination.requires_grad () == false);
            REQUIRE (grad == std::vector<int16_t> (4u, 0)); /* no lane stream */

            /* the same blob restored into a differently shaped view is refused */
            std::vector<int8_t> wide (5u, 0);
            Trainable wrong = Trainable::make (wide.data (), nullptr,
                                               Shape::vector (5u), 0u);
            REQUIRE (wck_restore (view, 0u, wrong) == Status::BAD_LAYOUT);
            REQUIRE (wide[0] == 0u);
        }

    SECTION ("the multiplier table copies into a model-sized buffer")
        {
            REQUIRE (status_ok (wck_read (blob.data (), 116u, view)));
            std::vector<uint8_t> out (8u, 0xEEu);
            uint32_t bytes_out = 0u;
            REQUIRE (status_ok (wck_copy_scale_table (view, 0u, out.data (), 8u,
                                                     bytes_out)));
            REQUIRE (bytes_out == 4u);
            REQUIRE (static_cast<int32_t> (rd32 (out.data ())) == -20000);
            REQUIRE (out[4] == 0xEEu);
            REQUIRE (wck_copy_scale_table (view, 0u, out.data (), 3u, bytes_out)
                    == Status::TRUNCATED);
            REQUIRE (bytes_out == 0u);
            REQUIRE (wck_copy_scale_table (view, 0u, out.data () + 1u, 8u, bytes_out)
                    == Status::BAD_LAYOUT);
            REQUIRE (wck_copy_scale_table (view, 0u, nullptr, 8u, bytes_out)
                    == Status::BAD_LAYOUT);
            REQUIRE (wck_copy_scale_table (view, 1u, out.data (), 8u, bytes_out)
                    == Status::OUT_OF_RANGE);
        }

    SECTION ("an invalid view makes every accessor refuse")
        {
            WckView broken;
            REQUIRE (wck_read (blob.data (), 100u, broken) == Status::TRUNCATED);
            REQUIRE (broken._M_valid == 0u);
            WckEntry entry;
            REQUIRE (wck_entry (broken, 0u, entry) == Status::BAD_LAYOUT);
            const uint8_t *codes = nullptr;
            uint32_t length = 0u;
            REQUIRE (wck_code_span (broken, 0u, codes, length) == Status::BAD_LAYOUT);
            int32_t value = 0;
            REQUIRE (wck_scale_at (broken, 0u, 0u, value) == Status::BAD_LAYOUT);
            uint32_t restored_out = 0u;
            uint32_t failed_out = 0u;
            std::vector<int8_t> code (4u, 0);
            std::vector<Trainable> list;
            list.push_back (Trainable::make (code.data (), nullptr,
                                             Shape::vector (4u), 0u));
            REQUIRE (wck_restore_set (broken, list.data (), 1u, restored_out,
                                     failed_out)
                     == Status::BAD_LAYOUT);
            Trainable destination
                = Trainable::make (code.data (), nullptr, Shape::vector (4u), 0u);
            REQUIRE (wck_restore (broken, 0u, destination) == Status::BAD_LAYOUT);
        }
}

TEST_CASE ("wck_read rejects a malformed container in the documented order")
{
    const WckFixture fixture;
    WckWriteStats stats;
    const std::vector<uint8_t> clean = write_wck (fixture, stats);
    const uint32_t total = static_cast<uint32_t> (clean.size ());
    REQUIRE (total == 208u);

    const auto read = [&] (const std::vector<uint8_t> &buf, uint32_t length) {
        WckView view;
        const Status status = wck_read (buf.data (), length, view);
        if (!status_ok (status))
            REQUIRE (view._M_valid == 0u);
        return status;
    };

    SECTION ("null, misaligned and below-header buffers")
        {
            REQUIRE (read (clean, 0u) == Status::TRUNCATED);
            REQUIRE (read (clean, 55u) == Status::TRUNCATED);
            WckView view;
            REQUIRE (wck_read (nullptr, total, view) == Status::BAD_LAYOUT);
            REQUIRE (wck_read (clean.data () + 1u, total - 1u, view)
                    == Status::BAD_LAYOUT);
            REQUIRE (wck_read (clean.data () + 3u, total - 3u, view)
                    == Status::BAD_LAYOUT);
        }

    SECTION ("magic then version then header size then the reserved word")
        {
            std::vector<uint8_t> buf = clean;
            buf[2] = static_cast<uint8_t> ('X');
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_MAGIC);

            buf = clean;
            buf[0] = static_cast<uint8_t> ('x');
            REQUIRE (read (buf, total) == Status::BAD_MAGIC); /* before the CRC */

            buf = clean;
            buf[4] = 2u;
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_VERSION);

            buf = clean;
            wr16 (buf.data () + 44, 60u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 52, 1u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("a header whose own CRC does not match is CORRUPT_STREAM")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 36, 9999u); /* step: structural rules still pass */
            REQUIRE (read (buf, total) == Status::CORRUPT_STREAM);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::OK);
            REQUIRE (rd16 (buf.data () + 48) != rd16 (clean.data () + 48));
        }

    SECTION ("parameter count limits")
        {
            std::vector<uint8_t> buf = clean;
            wr16 (buf.data () + 6, 0u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr16 (buf.data () + 6, static_cast<uint16_t> (_S_wck_max_params + 1u));
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("the table size must be the count times 48")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 8, 48u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("presence flags must agree with the region sizes")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 12, 0u); /* scale bytes gone, flag still set */
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            buf[5] = static_cast<uint8_t> (buf[5] & ~static_cast<uint8_t> (
                                              WckFlag::WckHasGrad));
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            buf[5] = static_cast<uint8_t> (buf[5] | static_cast<uint8_t> (
                                              WckFlag::WckHasScaleTable));
            wr32 (buf.data () + 12, 0u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 24, 0u); /* opt bytes gone, flag still set */
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("the payload and total sums are recomputed, not trusted")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 28, 57u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 32, 209u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 32, 209u); /* total no longer equals the sum */
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* a declared total that does exceed the geometry but not the buffer
               still reaches the length rule */
            REQUIRE (read (clean, 207u) == Status::TRUNCATED);
            REQUIRE (read (clean, 152u) == Status::TRUNCATED);
        }

    SECTION ("the payload's first word must be the documented zero")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 152u, 1u);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("a corrupted payload is CORRUPT_STREAM before any entry is read")
        {
            std::vector<uint8_t> buf = clean;
            buf[180u] = static_cast<uint8_t> (buf[180u] ^ 0xFFu);
            REQUIRE (read (buf, total) == Status::CORRUPT_STREAM);
            reseal_wck (buf);
            REQUIRE (read (buf, total) == Status::OK);
        }

    SECTION ("entries must tile their regions in payload order")
        {
            /* code offset of tensor 1 relocated */
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 56u + 48u + 20u, 29u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* gradient offset of tensor 0 moved by the alignment it should use */
            buf = clean;
            wr32 (buf.data () + 56u + 24u, 33u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* optimizer region dropped while its lanes survive */
            buf = clean;
            wr32 (buf.data () + 56u + 28u, 0u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* lanes claimed with no region */
            buf = clean;
            wr32 (buf.data () + 56u + 40u, 0u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* multiplier entries removed from a table the header still sizes */
            buf = clean;
            wr16 (buf.data () + 56u + 48u + 36u, 2u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* a table offset claimed with no entries */
            buf = clean;
            wr32 (buf.data () + 56u + 32u, 4u);
            wr16 (buf.data () + 56u + 36u, 0u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            /* the header's multiplier total disagrees with the entries */
            buf = clean;
            wr16 (buf.data () + 50, 3u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }

    SECTION ("entry shape, count and pad words are all validated")
        {
            std::vector<uint8_t> buf = clean;
            wr32 (buf.data () + 56u + 12u, 4u); /* count vs dims (5,1,1,1) */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr16 (buf.data () + 56u + 0u, 0u); /* a zero-length axis */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            buf[56u + 8u] = 0u; /* rank 0 */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            buf[56u + 8u] = 5u; /* rank past _S_max_rank */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            buf[56u + 9u] = 1u; /* pad byte must stay zero */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr16 (buf.data () + 56u + 38u, 1u); /* pad2 */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 56u + 44u, 7u); /* entry reserved */
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);

            buf = clean;
            wr32 (buf.data () + 56u + 12u, _S_max_elements + 1u);
            REQUIRE (read (buf, total) == Status::BAD_LAYOUT);
        }
}

TEST_CASE ("weight container corruption sweep pins exactly which bytes are covered")
{
    const WckFixture fixture;
    WckWriteStats stats;
    const std::vector<uint8_t> clean = write_wck (fixture, stats);
    const uint32_t total = static_cast<uint32_t> (clean.size ());
    const uint32_t payload_off = 152u;

    /* Table fields with no structural rule: the parameter table sits outside
       both CRC regions (the header CRC covers [0,48), the payload CRC covers
       the payload only), so these bytes per entry survive a flip unchallenged:
       the dimension words past the rank, the tensor flags and the tensor
       scale.  The rank byte itself is at entry offset 8, so the ignored dims
       are the bytes from 2 * rank to 7. */
    const auto table_exempt = [&clean](uint32_t offset) noexcept {
        if (offset < 56u || offset >= payload_off)
            return false;
        const uint32_t index
            = (offset - 56u) / static_cast<uint32_t> (_S_wck_entry_bytes);
        const uint32_t in_entry
            = (offset - 56u) % static_cast<uint32_t> (_S_wck_entry_bytes);
        const uint32_t rank
            = clean[56u + index * _S_wck_entry_bytes + 8u];
        return (in_entry >= 2u * rank && in_entry < 8u) /* dims past rank */
               || in_entry == 10u || in_entry == 11u /* flags */
               || (in_entry >= 16u && in_entry < 20u); /* scale_q15 */
    };

    const auto expected = [&](uint32_t offset) noexcept {
        if (offset < 4u)
            return Status::BAD_MAGIC;
        if (offset == 4u)
            return Status::BAD_VERSION;
        if (offset >= 44u && offset < 46u)
            return Status::BAD_LAYOUT; /* header size word */
        if (offset >= 50u && offset < 52u)
            return Status::BAD_LAYOUT; /* multiplier total, caught by the scan */
        if (offset >= 52u && offset < 56u)
            return Status::BAD_LAYOUT; /* header reserved word */
        if (offset < 56u)
            return Status::CORRUPT_STREAM; /* inside the header CRC */
        if (offset < payload_off)
            return table_exempt (offset) ? Status::OK : Status::BAD_LAYOUT;
        if (offset < payload_off + _S_wck_reserved_bytes)
            return Status::BAD_LAYOUT; /* payload reserved word */
        return Status::CORRUPT_STREAM; /* inside the payload CRC */
    };

    uint32_t caught = 0u;
    uint32_t exempt = 0u;
    for (uint32_t offset = 0u; offset < total; ++offset)
        {
            std::vector<uint8_t> buf = clean;
            buf[offset] = static_cast<uint8_t> (buf[offset] ^ 0x01u);
            WckView view;
            const Status status = wck_read (buf.data (), total, view);
            INFO ("flipped byte offset " << offset);
            REQUIRE (status == expected (offset));
            if (status == Status::OK)
                ++exempt;
            else
                ++caught;
        }
    REQUIRE (caught + exempt == total);
    /* Two rank-1 entries: six ignored dim bytes, two flag bytes and a 32-bit
       scale each, so twelve unprotected bytes per entry. */
    REQUIRE (exempt == 2u * 12u);
}

TEST_CASE ("the unprotected container table window is a real import hazard")
{
    const WckFixture fixture;
    WckWriteStats stats;
    std::vector<uint8_t> blob = write_wck (fixture, stats);

    /* A tampered tensor scale or flag in the parameter table is imported
       silently: no CRC covers it and no rule checks it.  This is the observable
       consequence, asserted so the gap is on record. */
    wr32 (blob.data () + 56u + 16u, 1u);           /* _M_scale_q15 := 1 */
    wr16 (blob.data () + 56u + 10u, 0u);           /* _M_flags := none */
    WckView view;
    REQUIRE (status_ok (wck_read (blob.data (),
                                  static_cast<uint32_t> (blob.size ()), view)));
    WckEntry entry;
    REQUIRE (status_ok (wck_entry (view, 0u, entry)));
    REQUIRE (entry._M_scale_q15 == 1);
    REQUIRE (entry._M_flags == 0u);

    std::vector<int8_t> code (5u, 0);
    std::vector<int16_t> grad (5u, 0);
    std::vector<Trainable> list;
    list.reserve (2u);
    list.push_back (Trainable::make (code.data (), grad.data (), Shape::vector (5u),
                                    Flag::RequiresGrad));
    REQUIRE (status_ok (wck_restore (view, 0u, list[0])));
    REQUIRE (list[0]._M_scale_q15 == 1);
    REQUIRE (list[0].requires_grad () == false);
    REQUIRE (code == fixture.a_code); /* the payload itself is still trusted */
}

TEST_CASE ("wck_validate_sources rejects every source list the reader would refuse")
{
    const WckFixture fixture;
    WckWriteStats stats;
    const std::vector<uint8_t> clean = write_wck (fixture, stats);
    WckView view;
    REQUIRE (status_ok (wck_read (clean.data (),
                                  static_cast<uint32_t> (clean.size ()), view)));

    SECTION ("the source array itself")
        {
            WckWriteStats local;
            std::vector<uint8_t> buf (512u, 0u);
            REQUIRE (wck_bytes_needed (nullptr, 1u, local._M_bytes_written)
                     == Status::BAD_LAYOUT);
            REQUIRE (wck_bytes_needed (fixture.sources.data (), 0u,
                                      local._M_bytes_written)
                     == Status::OUT_OF_RANGE);
            REQUIRE (wck_bytes_needed (fixture.sources.data () + 1u, 0u,
                                      local._M_bytes_written)
                     == Status::OUT_OF_RANGE);
            REQUIRE (wck_bytes_needed (fixture.sources.data (), 1025u,
                                      local._M_bytes_written)
                     == Status::OUT_OF_RANGE);
            REQUIRE (wck_write (buf.data (), 512u, nullptr, 1u, 0u, 0, local)
                    == Status::BAD_LAYOUT);
            REQUIRE (wck_write (buf.data (), 512u, fixture.sources.data (), 0u, 0u,
                               0, local)
                     == Status::OUT_OF_RANGE);
            REQUIRE (wck_write (buf.data () + 1u, 511u, fixture.sources.data (),
                                2u, 0u, 0, local)
                     == Status::BAD_LAYOUT);
        }

    SECTION ("a null or invalid parameter view")
        {
            std::vector<WckTensorSource> sources (1u);
            uint32_t needed = 0u;
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
            ThreeParams local;
            sources[0]._M_param = &local.views[0];
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 1u, needed)));
            sources[0]._M_param = nullptr;
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
            local.views[0]._M_count = 11u; /* shape/count disagree */
            sources[0]._M_param = &local.views[0];
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
            local.views[0]._M_count = 12u;
            local.views[0]._M_code = nullptr;
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
        }

    SECTION ("unaligned streams and a flag word wider than the entry field")
        {
            std::vector<int8_t> code (8u, 1);
            std::vector<int16_t> grad (8u, 0);
            std::vector<Trainable> views;
            views.push_back (Trainable::make (code.data () + 1u, grad.data (),
                                              Shape::vector (4u), 0u));
            std::vector<WckTensorSource> sources (1u);
            uint32_t needed = 0u;
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT); /* null param wins */
            sources[0]._M_param = &views[0];
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
            views[0] = Trainable::make (code.data (), grad.data () + 1u,
                                        Shape::vector (4u), 0u);
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::BAD_LAYOUT);
            views[0] = Trainable::make (code.data (), grad.data (),
                                        Shape::vector (4u), 0u);
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 1u, needed)));
            views[0]._M_flags = 0x10000u; /* does not fit the 16-bit entry field */
            REQUIRE (wck_bytes_needed (sources.data (), 1u, needed)
                    == Status::OUT_OF_RANGE);
        }

    SECTION ("optimizer lanes without a buffer, and vice versa")
        {
            std::vector<WckTensorSource> sources
                = fixture.sources;
            uint32_t needed = 0u;
            sources[0]._M_opt_lanes = 0u; /* buffer present, nothing to write */
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_opt = nullptr;
            sources[0]._M_opt_lanes = 5u; /* lanes claimed, no buffer */
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_opt_lanes = 0u;
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 2u, needed)));
            sources[0]._M_opt = fixture.a_opt.data ();
            sources[0]._M_opt_lanes = 4u * _S_max_elements + 1u;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::OUT_OF_RANGE);
            sources[0]._M_opt_lanes = 5u;
            sources[0]._M_opt = fixture.a_opt.data () + 1u;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
        }

    SECTION ("multiplier tables must match the axis they claim to walk")
        {
            std::vector<WckTensorSource> sources = fixture.sources;
            uint32_t needed = 0u;
            sources[0]._M_scale_entries = 0u;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::OUT_OF_RANGE);
            sources[0]._M_scale_entries = 1u;
            sources[0]._M_scale_axis = 0u;
            sources[0]._M_scale_table = fixture.a_scale.data ();
            /* Axis 0 of vector(5) is five lanes, so a one-entry table there is
               a shape mismatch, not a per-tensor shorthand. */
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_entries = 5u;
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 2u, needed)));
            sources[0]._M_scale_entries = 2u; /* per-tensor with two entries */
            sources[0]._M_scale_axis = _S_wck_per_tensor;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_entries = 1u;
            sources[0]._M_scale_axis = _S_max_rank; /* axis 4 does not exist */
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_axis = 1u; /* dim 1 of vector(5) is 1, not 5 */
            sources[0]._M_scale_entries = 2u;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_entries = 1u;
            sources[0]._M_scale_axis = _S_wck_per_tensor;
            sources[0]._M_scale_table
                = reinterpret_cast<const int32_t *> (
                      reinterpret_cast<const uint8_t *> (
                          fixture.a_scale.data ()) + 1u);
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_table = nullptr;
            sources[0]._M_scale_entries = 1u;
            REQUIRE (wck_bytes_needed (sources.data (), 2u, needed)
                    == Status::BAD_LAYOUT);
            sources[0]._M_scale_entries = 0u;
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 2u, needed)));
        }

    SECTION ("the multiplier table total is bounded by the header field")
        {
            std::vector<int8_t> code (30000u, 3);
            std::vector<int32_t> table (30000u, 40);
            std::vector<Trainable> views;
            views.push_back (Trainable::make (code.data (), nullptr,
                                             Shape::vector (30000u), 0u));
            std::vector<WckTensorSource> sources (3u);
            for (uint32_t i = 0u; i < 3u; ++i)
                {
                    sources[i]._M_param = &views[0];
                    sources[i]._M_scale_table = table.data ();
                    sources[i]._M_scale_entries = 30000u;
                    sources[i]._M_scale_axis = 0u;
                }
            uint32_t needed = 0u;
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 2u, needed)));
            REQUIRE (needed == _S_wck_header_bytes + 2u * _S_wck_entry_bytes
                              + _S_wck_reserved_bytes + 60000u
                              + 60000u * _S_wck_scale_entry_bytes);
            REQUIRE (wck_bytes_needed (sources.data (), 3u, needed)
                    == Status::OUT_OF_RANGE);
            REQUIRE (needed == 0u);
        }

    SECTION ("a payload past the 8 MiB ceiling cannot be sized or written")
        {
            std::vector<int8_t> code (_S_max_elements, 3);
            std::vector<Trainable> views;
            views.push_back (Trainable::make (code.data (), nullptr,
                                             Shape::vector (_S_max_elements), 0u));
            std::vector<WckTensorSource> sources (1024u);
            for (WckTensorSource &source : sources)
                source._M_param = &views[0];
            uint32_t needed = 0u;
            REQUIRE (wck_bytes_needed (sources.data (), 1024u, needed)
                    == Status::OUT_OF_RANGE);
            WckWriteStats local;
            std::vector<uint8_t> buf (1024u, 0u);
            REQUIRE (wck_write (buf.data (), 1024u, sources.data (), 1024u, 0u, 0,
                               local)
                     == Status::OUT_OF_RANGE);
            REQUIRE (local._M_bytes_written == 0u);
            REQUIRE (status_ok (wck_bytes_needed (sources.data (), 100u, needed)));
            REQUIRE (needed == _S_wck_header_bytes + 100u * _S_wck_entry_bytes
                              + _S_wck_reserved_bytes
                              + 100u * ref_align4 (_S_max_elements));
        }

    SECTION ("bytes needed is exactly what a successful write reports")
        {
            uint32_t needed = 0u;
            REQUIRE (status_ok (wck_bytes_needed (fixture.sources.data (), 2u,
                                                 needed)));
            REQUIRE (needed == static_cast<uint32_t> (clean.size ()));
            REQUIRE (needed == stats._M_bytes_written);
            REQUIRE (needed == _S_wck_header_bytes + 96u + 56u);
        }
}

TEST_CASE ("wck_restore_set walks a whole parameter set and names a failure")
{
    const WckFixture fixture;
    WckWriteStats stats;
    const std::vector<uint8_t> blob = write_wck (fixture, stats);
    WckView view;
    REQUIRE (status_ok (wck_read (blob.data (),
                                  static_cast<uint32_t> (blob.size ()), view)));

    SECTION ("a matching set restores codes, lanes, scales and flags")
        {
            std::vector<int8_t> a_code (5u, 0), b_code (3u, 0);
            std::vector<int16_t> a_grad (5u, 0);
            std::vector<Trainable> set;
            set.reserve (2u);
            set.push_back (Trainable::make (a_code.data (), a_grad.data (),
                                            Shape::vector (5u), 0u));
            set.push_back (Trainable::make (b_code.data (), nullptr,
                                            Shape::vector (3u), 0u));
            uint32_t restored_out = 0u;
            uint32_t failed_out = 0u;
            REQUIRE (status_ok (wck_restore_set (view, set.data (), 2u, restored_out,
                                                 failed_out)));
            REQUIRE (restored_out == 2u);
            REQUIRE (failed_out == 0u);
            REQUIRE (a_code == fixture.a_code);
            REQUIRE (b_code == fixture.b_code);
            REQUIRE (a_grad == fixture.a_grad);
            REQUIRE (set[0]._M_scale_q15 == 16000);
            REQUIRE (set[1]._M_scale_q15 == 24000);
            REQUIRE (set[0]._M_flags == fixture.views[0]._M_flags);
            REQUIRE (set[1]._M_flags == fixture.views[1]._M_flags);
            REQUIRE (set[0].requires_grad ());
            REQUIRE (not set[1].requires_grad ());
        }

    SECTION ("a destination without lanes cannot receive a lane stream")
        {
            std::vector<int8_t> a_code (5u, 0);
            std::vector<Trainable> set;
            set.reserve (2u);
            set.push_back (Trainable::make (a_code.data (), nullptr,
                                            Shape::vector (5u), 0u));
            set.push_back (Trainable::make (a_code.data (), nullptr,
                                            Shape::vector (3u), 0u));
            uint32_t restored_out = 0u;
            uint32_t failed_out = 0u;
            REQUIRE (wck_restore_set (view, set.data (), 2u, restored_out, failed_out)
                    == Status::BAD_LAYOUT);
            /* the codes are copied before the lane check fires, so a partially
               imported tensor is what the caller keeps: recorded, not assumed */
            REQUIRE (restored_out == 0u);
            REQUIRE (failed_out == 0u);
            REQUIRE (a_code == fixture.a_code);
        }

    SECTION ("the count must match the blob and the first mismatch aborts")
        {
            std::vector<int8_t> a_code (5u, 0), b_code (5u, 0);
            std::vector<int16_t> a_grad (5u, 0);
            std::vector<Trainable> set;
            set.reserve (2u);
            set.push_back (Trainable::make (a_code.data (), a_grad.data (),
                                            Shape::vector (5u), 0u));
            set.push_back (Trainable::make (b_code.data (), nullptr,
                                            Shape::vector (5u), 0u));
            uint32_t restored_out = 0u;
            uint32_t failed_out = 0u;
            REQUIRE (wck_restore_set (view, set.data (), 3u, restored_out, failed_out)
                    == Status::OUT_OF_RANGE);
            REQUIRE (wck_restore_set (view, set.data (), 1u, restored_out, failed_out)
                    == Status::OUT_OF_RANGE);
            REQUIRE (wck_restore_set (view, nullptr, 2u, restored_out, failed_out)
                    == Status::BAD_LAYOUT);
            /* tensor 0 succeeds, tensor 1 has the wrong shape */
            REQUIRE (wck_restore_set (view, set.data (), 2u, restored_out, failed_out)
                    == Status::BAD_LAYOUT);
            REQUIRE (restored_out == 1u);
            REQUIRE (failed_out == 1u);
            REQUIRE (a_code == fixture.a_code);
            REQUIRE (b_code == std::vector<int8_t> (5u, 0));
        }

    SECTION ("a blob carrying -128 codes is imported raw, unlike copy_out")
        {
            /* the container is a transport, not a validator: only part A and the
               model bridge refuse the rail code, so a raw restore can land it */
            std::vector<uint8_t> tampered = blob;
            tampered[152u + 20u] = 0x80u; /* payload code byte of tensor 0 */
            reseal_wck (tampered);
            WckView local;
            REQUIRE (status_ok (wck_read (tampered.data (),
                                          static_cast<uint32_t> (tampered.size ()),
                                          local)));
            std::vector<int8_t> code (5u, 0);
            std::vector<int16_t> grad (5u, 0);
            Trainable destination = Trainable::make (code.data (), grad.data (),
                                                    Shape::vector (5u), 0u);
            REQUIRE (status_ok (wck_restore (local, 0u, destination)));
            REQUIRE (code[0] == -128);
            std::vector<uint8_t> model (12u, 0u);
            REQUIRE (copy_out (destination, model.data (), 12u, 0u, 5u)
                    == Status::CORRUPT_STREAM);
            REQUIRE (model == std::vector<uint8_t> (12u, 0u));
        }
}
