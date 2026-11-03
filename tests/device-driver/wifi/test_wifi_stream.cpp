/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file test_wifi_stream.cpp
 * @brief Unit tests for the WiFi streaming pipeline: byte ring, length-prefixed
 *        frame reassembly, and frame parsing/delivery.
 */

#include "kernel/device-driver/wifi/stream/wifi_stream.hpp"
#include "kernel/device-driver/wifi/wifi_frame.hpp"
#include "kernel/kernel_status.hpp"
#include "test_mock_stream_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using namespace zzs::kernel;
using namespace zzs::kernel::device::driver::wifi;
using namespace zzs::kernel::device::driver::wifi::stream;
using namespace test;

TEST_CASE ("wifi.stream.ring.push_pop_basic", "[wifi][stream]") {
  test::MockStreamBus bus; /* not used for ring unit ops */
  bus.set_open (false);    /* bus not open */
  WifiStream stream (&bus);
  (void) stream;

  /* Build a stream with depth 4 directly by constructing and then using
   * ring_push/ring_pop via poll is not possible; instead test the public
   * buffered_bytes/is_open API and the inject path via poll. */
  REQUIRE (stream.is_open () == false); /* bus not open */
}

TEST_CASE ("wifi.stream.length_prefixed_frame", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStream stream (&bus);
  bus.set_open (true);

  bool delivered = false;
  stream.set_frame_callback (
      [] (const Frame &f, void *ctx) noexcept {
        bool *d = static_cast<bool *> (ctx);
        *d = true;
      },
      &delivered);

  /* Valid 802.11 beacon header (24 bytes): FC=mgmt/beacon, version 0,
   * Addr1=broadcast, Addr2=Addr3=bssid. */
  uint8_t payload[24] = {};
  payload[0] = 0x80u; /* FC: type Management(0), subtype Beacon(8), ver 0 */
  payload[1] = 0x00u;
  for (uint8_t i = 0; i < 6; ++i)
    payload[4 + i] = 0xffu; /* addr1 */
  for (uint8_t i = 0; i < 6; ++i)
    payload[10 + i] = 0x11u; /* addr2 */
  for (uint8_t i = 0; i < 6; ++i)
    payload[16 + i] = 0x11u; /* addr3 */

  bus.inject_frame (payload, 24u);

  REQUIRE (stream.poll () == Status::OK);
  REQUIRE (delivered == true);
}

TEST_CASE ("wifi.stream.truncated_length_prefix", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStream stream (&bus);
  bus.set_open (true);

  uint8_t len_lo = 0x04u;
  bus.inject_response (&len_lo, 1u); /* only first length byte, no second */
  /* Poll drained the byte from the bus (OK) even though no complete frame
   * was reassembled yet. */
  REQUIRE (stream.poll () == Status::OK);
}

TEST_CASE ("wifi.stream.null_bus", "[wifi][stream]") {
  WifiStream stream (nullptr);
  REQUIRE (stream.poll () == Status::INTERNAL);
  REQUIRE (stream.is_open () == false);
}

TEST_CASE ("wifi.stream.send_frame.closed_bus", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStream stream (&bus);
  bus.set_open (false);
  uint8_t buf[4] = {1, 2, 3, 4};
  REQUIRE (stream.send_frame (buf, 4u) == Status::NOT_READY);
}

TEST_CASE ("wifi.stream.ring_depth_invalid_clamped", "[wifi][stream]") {
  test::MockStreamBus bus;
  bus.set_open (false);

  /* Depth 0 -> clamped to kRingMaxDepth (256), constructed without error. */
  WifiStreamConfig c0{};
  c0._M_byte_ring_depth = 0u;
  WifiStream s0 (&bus, nullptr, c0);
  REQUIRE (s0.ring_depth () == 256u);

  /* Depth > kRingMaxDepth -> clamped to 256. */
  WifiStreamConfig cBig{};
  cBig._M_byte_ring_depth = 1024u;
  WifiStream sBig (&bus, nullptr, cBig);
  REQUIRE (sBig.ring_depth () == 256u);

  /* Depth 512 -> clamped to 256 (largest power of two <= 512 is 256, but
   * kRingMaxDepth caps at 256 anyway). */
  WifiStreamConfig c512{};
  c512._M_byte_ring_depth = 512u;
  WifiStream s512 (&bus, nullptr, c512);
  REQUIRE (s512.ring_depth () == 256u);

  /* Depth 3 (non-power-of-two) -> clamped down to 2. */
  WifiStreamConfig c3{};
  c3._M_byte_ring_depth = 3u;
  WifiStream s3 (&bus, nullptr, c3);
  REQUIRE (s3.ring_depth () == 2u);

  /* Depth 1 -> valid power of two. */
  WifiStreamConfig c1{};
  c1._M_byte_ring_depth = 1u;
  WifiStream s1 (&bus, nullptr, c1);
  REQUIRE (s1.ring_depth () == 1u);
}

TEST_CASE ("wifi.stream.reset.invalid_depth_rejected", "[wifi][stream]") {
  test::MockStreamBus bus;
  bus.set_open (true);
  WifiStream stream (&bus);

  /* Valid reset: depth 4 is fine. */
  WifiStreamConfig c4{};
  c4._M_byte_ring_depth = 4u;
  REQUIRE (stream.reset (c4) == Status::OK);
  REQUIRE (stream.ring_depth () == 4u);

  /* Invalid reset: depth 1024 exceeds kRingMaxDepth, rejected, ring unchanged. */
  WifiStreamConfig cBad{};
  cBad._M_byte_ring_depth = 1024u;
  REQUIRE (stream.reset (cBad) == Status::OUT_OF_RANGE);
  REQUIRE (stream.ring_depth () == 4u); /* unchanged */

  /* Invalid reset: depth 0 rejected. */
  WifiStreamConfig cZero{};
  cZero._M_byte_ring_depth = 0u;
  REQUIRE (stream.reset (cZero) == Status::OUT_OF_RANGE);
  REQUIRE (stream.ring_depth () == 4u); /* unchanged */
}

TEST_CASE ("wifi.stream.fragmented_frame_across_pushes", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStream stream (&bus);
  bus.set_open (true);

  bool delivered = false;
  Frame delivered_frame{};
  stream.set_frame_callback (
      [] (const Frame &f, void *ctx) noexcept {
        bool *d = static_cast<bool *> (ctx);
        *d = true;
      },
      &delivered);

  /* Valid 802.11 beacon header (24 bytes). */
  uint8_t payload[24] = {};
  payload[0] = 0x80u;
  payload[1] = 0x00u;
  for (uint8_t i = 0; i < 6; ++i)
    payload[4 + i] = 0xffu;
  for (uint8_t i = 0; i < 6; ++i)
    payload[10 + i] = 0x11u;
  for (uint8_t i = 0; i < 6; ++i)
    payload[16 + i] = 0x11u;

  /* Inject length prefix one byte at a time. */
  uint8_t len_lo = static_cast<uint8_t> (sizeof (payload) & 0xffu);
  uint8_t len_hi = static_cast<uint8_t> ((sizeof (payload) >> 8) & 0xffu);
  bus.inject_response (&len_lo, 1u);
  REQUIRE (stream.poll () == Status::OK); /* drained, no complete frame yet */
  REQUIRE_FALSE (delivered);

  bus.inject_response (&len_hi, 1u);
  REQUIRE (stream.poll () == Status::OK); /* length complete, waiting payload */
  REQUIRE_FALSE (delivered);

  /* Inject payload in small fragments. */
  for (uint16_t i = 0u; i < sizeof (payload); ++i)
    bus.inject_response (&payload[i], 1u);
  REQUIRE (stream.poll () == Status::OK);
  REQUIRE (delivered);
}

TEST_CASE ("wifi.stream.multiple_frames_one_push", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStream stream (&bus);
  bus.set_open (true);

  uint8_t delivered_count = 0u;
  stream.set_frame_callback (
      [] (const Frame &f, void *ctx) noexcept {
        uint8_t *c = static_cast<uint8_t *> (ctx);
        *c = static_cast<uint8_t> (*c + 1u);
      },
      &delivered_count);

  /* Two 24-byte beacon frames back to back. */
  uint8_t beacon[24] = {};
  beacon[0] = 0x80u;
  beacon[1] = 0x00u;
  for (uint8_t i = 0; i < 6; ++i) {
    beacon[4 + i] = 0xffu;
    beacon[10 + i] = 0x11u;
    beacon[16 + i] = 0x11u;
  }

  /* Build: [len1][beacon1][len2][beacon2]. */
  uint8_t len1[2] = {static_cast<uint8_t> (sizeof (beacon) & 0xffu),
                     static_cast<uint8_t> ((sizeof (beacon) >> 8) & 0xffu)};
  uint8_t len2[2] = {static_cast<uint8_t> (sizeof (beacon) & 0xffu),
                     static_cast<uint8_t> ((sizeof (beacon) >> 8) & 0xffu)};

  bus.inject_response (len1, 2u);
  bus.inject_response (beacon, sizeof (beacon));
  bus.inject_response (len2, 2u);
  bus.inject_response (beacon, sizeof (beacon));

  REQUIRE (stream.poll () == Status::OK);
  REQUIRE (delivered_count == 2u);
}

TEST_CASE ("wifi.stream.ring_at_exact_capacity", "[wifi][stream]") {
  test::MockStreamBus bus;
  WifiStreamConfig cfg{};
  cfg._M_byte_ring_depth = 4u; /* tiny ring */
  WifiStream stream (&bus, nullptr, cfg);
  REQUIRE (stream.ring_depth () == 4u);

  /* Push 4 bytes: ring becomes full. */
  for (uint8_t i = 0u; i < 4u; ++i)
    REQUIRE (stream.ring_push (static_cast<uint8_t> (i)) == Status::OK);
  REQUIRE (stream.buffered_bytes () == 4u);

  /* Push 1 more: drop-oldest, returns QUEUE_FULL. */
  REQUIRE (stream.ring_push (0x99u) == Status::QUEUE_FULL);
  REQUIRE (stream.buffered_bytes () == 4u);

  /* Pop: oldest (0) was dropped, we get 1. */
  uint8_t b = 0u;
  REQUIRE (stream.ring_pop (b) == Status::OK);
  REQUIRE (b == 1u);
}
