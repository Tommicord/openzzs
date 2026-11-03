/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file i2c_bitbang_cpp.cpp
 * @brief Portable C++ I2C bit-bang fallback using Linux GPIO sysfs.
 *
 * Provides the same extern "C" bit-bang primitives as the assembly routines
 * in bus/asm/, implemented in plain C++ over the Linux GPIO sysfs interface
 * (/sys/class/gpio/gpioN/value, /direction).  Used when no assembly bit-bang
 * library is available for the target.
 *
 * No heap, no float, no RTTI, no exceptions.  All functions noexcept.
 */

#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

namespace zzs::kernel::sensor::driver::bus {

namespace {

/** @brief GPIO sysfs base path. */
constexpr char k_gpio_sysfs[] = "/sys/class/gpio";

/**
 * @brief Write a small text value to a GPIO sysfs attribute file.
 * @param path  full sysfs file path (e.g. /sys/class/gpio/gpio4/value)
 * @param value NUL-terminated string to write ("0", "1", "in", "out", ...)
 * @return 0 on success, -1 on error
 */
[[nodiscard]] int
sysfs_write (const char *path, const char *value) noexcept {
    int fd = open (path, O_WRONLY);
    if (fd < 0)
        return -1;
    ssize_t n = write (fd, value, std::strlen (value));
    int saved = (n < 0) ? -1 : 0;
    close (fd);
    return saved;
}

/**
 * @brief Read a small text value from a GPIO sysfs attribute file.
 * @param path     full sysfs file path
 * @param out      output buffer (at least 1 byte)
 * @param out_size size of output buffer
 * @return 0 on success, -1 on error
 */
[[nodiscard]] int
sysfs_read (const char *path, char *out, size_t out_size) noexcept {
    if (out_size == 0u)
        return -1;
    int fd = open (path, O_RDONLY);
    if (fd < 0)
        return -1;
    ssize_t n = read (fd, out, out_size - 1u);
    int saved = (n < 0) ? -1 : 0;
    if (n > 0)
        out[static_cast<size_t> (n)] = '\0';
    else
        out[0] = '\0';
    close (fd);
    return saved;
}

/**
 * @brief Build a GPIO sysfs path into a fixed buffer.
 * @param pin      GPIO pin number
 * @param attr     attribute name ("value", "direction", ...)
 * @param buf      output buffer
 * @param buf_size size of output buffer
 * @return pointer into buf, or nullptr if truncated
 */
[[nodiscard]] const char *
gpio_path (uint32_t pin, const char *attr, char *buf, size_t buf_size) noexcept {
    int n = std::snprintf (buf, buf_size, "%s/gpio%u/%s", k_gpio_sysfs, pin, attr);
    if (n < 0 || static_cast<size_t> (n) >= buf_size)
        return nullptr;
    return buf;
}

/** @brief Export a GPIO pin via sysfs (idempotent: ignores errors). */
[[nodiscard]] int
gpio_export (uint32_t pin) noexcept {
    char export_path[64];
    std::snprintf (export_path, sizeof (export_path), "%s/export", k_gpio_sysfs);
    char pin_str[16];
    std::snprintf (pin_str, sizeof (pin_str), "%u", pin);
    /* Ignore failure: pin may already be exported. */
    (void) sysfs_write (export_path, pin_str);
    return 0;
}

/** @brief Set GPIO pin direction via sysfs. */
[[nodiscard]] int
gpio_direction (uint32_t pin, const char *direction) noexcept {
    char path[96];
    const char *p = gpio_path (pin, direction, path, sizeof (path));
    if (p == nullptr)
        return -1;
    return sysfs_write (p, direction);
}

/** @brief Set GPIO pin value via sysfs. */
[[nodiscard]] int
gpio_set_value (uint32_t pin, uint8_t value) noexcept {
    char path[96];
    const char *p = gpio_path (pin, "value", path, sizeof (path));
    if (p == nullptr)
        return -1;
    return sysfs_write (p, value != 0u ? "1" : "0");
}

/** @brief Get GPIO pin value via sysfs. Returns 0 or 1, or -1 on error. */
[[nodiscard]] int
gpio_get_value (uint32_t pin) noexcept {
    char path[96];
    const char *p = gpio_path (pin, "value", path, sizeof (path));
    if (p == nullptr)
        return -1;
    char buf[8];
    if (sysfs_read (p, buf, sizeof (buf)) != 0)
        return -1;
    return (buf[0] == '1') ? 1 : 0;
}

/** @brief Best-effort microsecond delay using nanosleep. */
void
delay_us_cpp (uint32_t us) noexcept {
    if (us == 0u)
        return;
    timespec ts;
    ts.tv_sec = static_cast<time_t> (us / 1000000u);
    ts.tv_nsec = static_cast<long> ((us % 1000000u) * 1000L);
    while (nanosleep (&ts, &ts) != 0) {
        /* EINTR: retry with remaining time already in ts. */
    }
}

} // namespace

/* ------------------------------------------------------------------ */
/* extern "C" bit-bang primitives (C++ sysfs fallback)                */
/* ------------------------------------------------------------------ */

extern "C" {

void
bus_gpio_set (uint32_t /*gpio_base*/, uint32_t pin, uint8_t value) noexcept {
    (void) gpio_export (pin);
    (void) gpio_direction (pin, "out");
    (void) gpio_set_value (pin, value);
}

uint8_t
bus_gpio_get (uint32_t /*gpio_base*/, uint32_t pin) noexcept {
    (void) gpio_export (pin);
    (void) gpio_direction (pin, "in");
    int v = gpio_get_value (pin);
    return static_cast<uint8_t> (v < 0 ? 0 : v);
}

void
bus_i2c_delay_us (uint32_t us) noexcept {
    delay_us_cpp (us);
}

void
bus_i2c_start (uint32_t /*gpio_base*/, uint32_t sda_pin, uint32_t scl_pin) noexcept {
    /* Start condition: SDA goes low while SCL is high. */
    (void) gpio_export (sda_pin);
    (void) gpio_export (scl_pin);
    (void) gpio_direction (scl_pin, "out");
    (void) gpio_direction (sda_pin, "out");
    (void) gpio_set_value (sda_pin, 1u); /* SDA high while SCL high */
    (void) gpio_set_value (scl_pin, 1u); /* SCL high */
    bus_i2c_delay_us (4u);
    (void) gpio_set_value (sda_pin, 0u); /* SDA low -> start */
    bus_i2c_delay_us (4u);
}

void
bus_i2c_stop (uint32_t /*gpio_base*/, uint32_t sda_pin, uint32_t scl_pin) noexcept {
    /* Stop condition: SDA goes high while SCL is high. */
    (void) gpio_direction (sda_pin, "out");
    (void) gpio_direction (scl_pin, "out");
    (void) gpio_set_value (sda_pin, 0u); /* SDA low while SCL high */
    (void) gpio_set_value (scl_pin, 1u); /* SCL high */
    bus_i2c_delay_us (4u);
    (void) gpio_set_value (sda_pin, 1u); /* SDA high -> stop */
    bus_i2c_delay_us (4u);
}

uint8_t
bus_i2c_write_bit (uint32_t /*gpio_base*/, uint32_t sda_pin, uint32_t scl_pin, uint8_t bit) noexcept {
    /* Drive SDA according to bit. */
    (void) gpio_direction (sda_pin, "out");
    (void) gpio_set_value (sda_pin, bit != 0u ? 1u : 0u);
    bus_i2c_delay_us (2u);
    /* Pulse SCL high. */
    (void) gpio_direction (scl_pin, "out");
    (void) gpio_set_value (scl_pin, 1u);
    bus_i2c_delay_us (2u);
    /* Sample SDA while SCL high (ACK/NACK from slave). */
    (void) gpio_direction (sda_pin, "in");
    bus_i2c_delay_us (1u);
    uint8_t ack = static_cast<uint8_t> (gpio_get_value (sda_pin));
    (void) gpio_set_value (scl_pin, 0u); /* SCL low */
    bus_i2c_delay_us (2u);
    return ack; /* 0 = ACK, 1 = NACK */
}

uint8_t
bus_i2c_read_bit (uint32_t /*gpio_base*/, uint32_t sda_pin, uint32_t scl_pin) noexcept {
    /* Release SDA for read. */
    (void) gpio_direction (sda_pin, "in");
    bus_i2c_delay_us (2u);
    /* Pulse SCL high. */
    (void) gpio_direction (scl_pin, "out");
    (void) gpio_set_value (scl_pin, 1u);
    bus_i2c_delay_us (2u);
    uint8_t bit = static_cast<uint8_t> (gpio_get_value (sda_pin));
    (void) gpio_set_value (scl_pin, 0u); /* SCL low */
    bus_i2c_delay_us (2u);
    return bit;
}

uint8_t
bus_i2c_write_byte (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin, uint8_t byte) noexcept {
    /* Write MSB-first, 8 bits, then read ACK/NACK. */
    for (int32_t bit = 7; bit >= 0; --bit) {
        uint8_t b = static_cast<uint8_t> ((byte >> bit) & 1u);
        (void) bus_i2c_write_bit (gpio_base, sda_pin, scl_pin, b);
    }
    /* Read ACK/NACK after 8 bits. */
    return bus_i2c_read_bit (gpio_base, sda_pin, scl_pin);
}

uint8_t
bus_i2c_read_byte (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin, uint8_t ack) noexcept {
    /* Read 8 bits MSB-first. */
    uint8_t val = 0u;
    for (int32_t bit = 7; bit >= 0; --bit) {
        uint8_t b = bus_i2c_read_bit (gpio_base, sda_pin, scl_pin);
        val = static_cast<uint8_t> ((val << 1) | b);
    }
    /* Send ACK (ack==0) or NACK (ack!=0). */
    (void) bus_i2c_write_bit (gpio_base, sda_pin, scl_pin, ack != 0u ? 1u : 0u);
    return val;
}

} // extern "C"

} // namespace zzs::kernel::sensor::driver::bus
