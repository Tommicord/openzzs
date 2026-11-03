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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#include "platform_hal.hpp"

#ifdef __QNX__
#include <fcntl.h>
#include <hw/i2c.h>
#include <hw/spi.h>
#include <sys/iocmd.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"
#include "kernel/sensor-driver/bus/spi/spi_bitbang.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "std/fmt.hpp"

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::qnx {

namespace {

/** @brief QNX /dev/i2c<N> bus adapter (devctl DCMD_I2C). */
class QnxI2cBus : public bus::BusInterface {
  public:
    QnxI2cBus (int bus_num, uint8_t addr) noexcept : _M_fd (-1), _M_addr (addr) {
#ifdef __QNX__
        char path[32];
        const unsigned int n = zzs::base::format_into (path, "/dev/i2c{}", bus_num);
        (void) n;
        _M_fd = open (path, O_RDWR);
#endif
    }
    ~QnxI2cBus () noexcept {
#ifdef __QNX__
        if (_M_fd >= 0)
            close (_M_fd);
#endif
    }
    QnxI2cBus (const QnxI2cBus &) = delete;
    QnxI2cBus &
    operator= (const QnxI2cBus &) = delete;
    QnxI2cBus (QnxI2cBus &&) = delete;
    QnxI2cBus &
    operator= (QnxI2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef __QNX__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        i2c_send_t snd = {};
        snd.addr = _M_addr;
        snd.cnt = static_cast<uint32_t> (len + 1u);
        snd.stop = 1;
        snd.data = buf;
        if (devctl (_M_fd, DCMD_I2C_SEND, &snd, sizeof (snd), nullptr) != EOK)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef __QNX__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        i2c_send_t snd = {};
        snd.addr = _M_addr;
        snd.cnt = 1u;
        snd.stop = 0;
        snd.data = &reg;
        if (devctl (_M_fd, DCMD_I2C_SEND, &snd, sizeof (snd), nullptr) != EOK)
            return zzs::kernel::Status::INTERNAL;
        i2c_recv_t rcv = {};
        rcv.addr = _M_addr;
        rcv.cnt = len;
        rcv.stop = 1;
        rcv.data = data;
        if (devctl (_M_fd, DCMD_I2C_RECV, &rcv, sizeof (rcv), nullptr) != EOK)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef __QNX__
        struct timespec ts = {ms / 1000, (ms % 1000) * 1000000ull};
        clock_nanosleep (CLOCK_MONOTONIC, 0, &ts, nullptr);
#else
        (void) ms;
#endif
    }

  private:
    int _M_fd;
    uint8_t _M_addr;
};

/** @brief Monotonic nanoseconds via ClockTime(CLOCK_MONOTONIC). */
[[nodiscard]] uint64_t
qnx_now_ns (void) noexcept {
#ifdef __QNX__
    struct timespec ts;
    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
        return static_cast<uint64_t> (ts.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts.tv_nsec);
#endif
    return 0ull;
}

static QnxI2cBus i2c_accel_bus (1, 0x6Au);
static QnxI2cBus i2c_mag_bus (1, 0x1Cu);
static QnxI2cBus i2c_light_bus (1, 0x39u);

static sensors::Lsm6dsoAccelDriver accel_driver (&i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver gyro_driver (&i2c_accel_bus);
static sensors::Lis3mdlMagDriver mag_driver (&i2c_mag_bus);
static sensors::Tsl2561LightDriver light_driver (&i2c_light_bus);
static sensors::Ics43434SoundDriver sound_driver (&i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal hal_vtable;

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept {
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries) {
        switch (idx) {
        case 0u:
            st = accel_driver.read_sample (out);
            break;
        case 1u:
            st = gyro_driver.read_sample (out);
            break;
        case 2u:
            st = mag_driver.read_sample (out);
            break;
        case 3u:
            st = light_driver.read_sample (out);
            break;
        default:
            st = sound_driver.read_sample (out);
            break;
        }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (zzs::kernel::status_ok (st))
            return zzs::kernel::Status::OK;
    }
    return st;
}

[[nodiscard]] zzs::kernel::Status
hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept {
    (void) accel_driver.set_channel_enabled (ch, enabled);
    (void) gyro_driver.set_channel_enabled (ch, enabled);
    (void) mag_driver.set_channel_enabled (ch, enabled);
    (void) light_driver.set_channel_enabled (ch, enabled);
    (void) sound_driver.set_channel_enabled (ch, enabled);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
hal_channel_rate_hz (sensor::Channel ch) noexcept {
    return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
hal_init (void) noexcept {
    zzs::kernel::Status st = clock_source_install (&qnx_now_ns);
    if (!zzs::kernel::status_ok (st))
        return st;
    st = accel_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = gyro_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = mag_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = light_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = sound_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    return zzs::kernel::Status::OK;
}

} // namespace

[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_qnx_sensor_hal () noexcept {
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::qnx
