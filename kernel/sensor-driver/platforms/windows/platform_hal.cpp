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

#ifdef _WIN32
#  include <windows.h>
#endif

#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"
#include "kernel/sensor-driver/bus/spi/spi_bitbang.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::windows
{

namespace {

/** @brief Platform-native Windows I2C bus adapter (CreateFile/DeviceIoControl). */
class WindowsI2cBus : public bus::BusInterface
{
public:
    explicit WindowsI2cBus (const wchar_t *device_path, uint8_t addr) noexcept
        : _M_handle (INVALID_HANDLE_VALUE)
        , _M_addr (addr)
    {
#ifdef _WIN32
        _M_handle = ::CreateFileW (device_path, GENERIC_READ | GENERIC_WRITE,
                                   0u, nullptr, OPEN_EXISTING, 0u, nullptr);
        if (_M_handle != INVALID_HANDLE_VALUE)
            {
            DWORD dw = 0;
            ::DeviceIoControl (_M_handle, IOCTL_I2C_SET_SLAVE_ADDRESS,
                               &_M_addr, sizeof (_M_addr), nullptr, 0, &dw, nullptr);
            }
#endif
    }

    ~WindowsI2cBus () noexcept
    {
#ifdef _WIN32
        if (_M_handle != INVALID_HANDLE_VALUE)
            ::CloseHandle (_M_handle);
#endif
    }

    WindowsI2cBus (const WindowsI2cBus &) = delete;
    WindowsI2cBus &operator= (const WindowsI2cBus &) = delete;
    WindowsI2cBus (WindowsI2cBus &&) = delete;
    WindowsI2cBus &operator= (WindowsI2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef _WIN32
        if (_M_handle == INVALID_HANDLE_VALUE || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        DWORD written = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_I2C_WRITE, buf,
                                static_cast<DWORD> (len + 1u), nullptr, 0,
                                &written, nullptr))
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef _WIN32
        if (_M_handle == INVALID_HANDLE_VALUE || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        DWORD written = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_I2C_WRITE, &reg, 1u,
                                nullptr, 0, &written, nullptr))
            return zzs::kernel::Status::INTERNAL;
        DWORD read = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_I2C_READ, nullptr, 0,
                                data, static_cast<DWORD> (len), &read, nullptr))
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef _WIN32
        ::Sleep (static_cast<DWORD> (ms));
#else
        (void) ms;
#endif
    }

private:
    void *_M_handle;
    uint8_t _M_addr;
};

/** @brief Platform-native Windows SPI bus adapter (CreateFile/DeviceIoControl). */
class WindowsSpiBus : public bus::BusInterface
{
public:
    explicit WindowsSpiBus (const wchar_t *device_path) noexcept
        : _M_handle (INVALID_HANDLE_VALUE)
        , _M_speed (1000000u)
    {
#ifdef _WIN32
        _M_handle = ::CreateFileW (device_path, GENERIC_READ | GENERIC_WRITE,
                                   0u, nullptr, OPEN_EXISTING, 0u, nullptr);
#endif
    }

    ~WindowsSpiBus () noexcept
    {
#ifdef _WIN32
        if (_M_handle != INVALID_HANDLE_VALUE)
            ::CloseHandle (_M_handle);
#endif
    }

    WindowsSpiBus (const WindowsSpiBus &) = delete;
    WindowsSpiBus &operator= (const WindowsSpiBus &) = delete;
    WindowsSpiBus (WindowsSpiBus &&) = delete;
    WindowsSpiBus &operator= (WindowsSpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef _WIN32
        if (_M_handle == INVALID_HANDLE_VALUE || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        DWORD written = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_SPI_WRITE, tx,
                                static_cast<DWORD> (len + 1u), nullptr, 0,
                                &written, nullptr))
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef _WIN32
        if (_M_handle == INVALID_HANDLE_VALUE || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[2] = { reg, 0 };
        DWORD written = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_SPI_WRITE, tx, 1u,
                                nullptr, 0, &written, nullptr))
            return zzs::kernel::Status::INTERNAL;
        DWORD read = 0;
        if (!::DeviceIoControl (_M_handle, IOCTL_SPI_READ, nullptr, 0,
                                data, static_cast<DWORD> (len), &read, nullptr))
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef _WIN32
        ::Sleep (static_cast<DWORD> (ms));
#else
        (void) ms;
#endif
    }

private:
    void *_M_handle;
    uint32_t _M_speed;
};

/** @brief Monotonic nanoseconds via QueryPerformanceCounter. */
[[nodiscard]] uint64_t
windows_now_ns (void) noexcept
{
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    if (::QueryPerformanceFrequency (&freq) && ::QueryPerformanceCounter (&cnt))
        return static_cast<uint64_t> (cnt.QuadPart) * 1000000000ull /
               static_cast<uint64_t> (freq.QuadPart);
#endif
    return 0ull;
}

static WindowsI2cBus i2c_accel_bus (L"\\\\.\\I2C1", 0x6Au);   // LSM6DSO
static WindowsI2cBus i2c_mag_bus (L"\\\\.\\I2C1", 0x1Cu);      // LIS3MDL
static WindowsI2cBus i2c_light_bus (L"\\\\.\\I2C1", 0x39u);    // TSL2561 / VEML7700

static sensors::Lsm6dsoAccelDriver accel_driver (&i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver gyro_driver (&i2c_accel_bus);
static sensors::Lis3mdlMagDriver mag_driver (&i2c_mag_bus);
static sensors::Tsl2561LightDriver light_driver (&i2c_light_bus);
static sensors::Ics43434SoundDriver sound_driver (&i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal hal_vtable;

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept
{
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries)
        {
        switch (idx)
            {
            case 0u: st = accel_driver.read_sample (out); break;
            case 1u: st = gyro_driver.read_sample (out); break;
            case 2u: st = mag_driver.read_sample (out); break;
            case 3u: st = light_driver.read_sample (out); break;
            default: st = sound_driver.read_sample (out); break;
            }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (zzs::kernel::status_ok (st))
            return zzs::kernel::Status::OK;
        }
    return st;
}

[[nodiscard]] zzs::kernel::Status
hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept
{
    (void) accel_driver.set_channel_enabled (ch, enabled);
    (void) gyro_driver.set_channel_enabled (ch, enabled);
    (void) mag_driver.set_channel_enabled (ch, enabled);
    (void) light_driver.set_channel_enabled (ch, enabled);
    (void) sound_driver.set_channel_enabled (ch, enabled);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
hal_channel_rate_hz (sensor::Channel ch) noexcept
{
    return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
hal_init (void) noexcept
{
    zzs::kernel::Status st = clock_source_install (&windows_now_ns);
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
create_windows_sensor_hal () noexcept
{
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::windows
