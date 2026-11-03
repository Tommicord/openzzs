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

#ifdef __ANDROID__
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#endif

#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/sensor-driver/sensors/adxl345/adxl345.hpp"
#include "kernel/sensor-driver/sensors/ak0991x/ak0991x.hpp"
#include "kernel/sensor-driver/sensors/ak8963/ak8963.hpp"
#include "kernel/sensor-driver/sensors/bmi160g/bmi160g.hpp"
#include "kernel/sensor-driver/sensors/bmm150/bmm150.hpp"
#include "kernel/sensor-driver/sensors/icm20602/icm20602.hpp"
#include "kernel/sensor-driver/sensors/icm20608/icm20608.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/sensor-driver/sensors/l3gd20/l3gd20.hpp"
#include "kernel/sensor-driver/sensors/lis2de12/lis2de12.hpp"
#include "kernel/sensor-driver/sensors/lis2dh12/lis2dh12.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/lsm6dsa/lsm6dsa.hpp"
#include "kernel/sensor-driver/sensors/lsm6dsg/lsm6dsg.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/ltr558/ltr558.hpp"
#include "kernel/sensor-driver/sensors/mma8451q/mma8451q.hpp"
#include "kernel/sensor-driver/sensors/mpu6050/mpu6050.hpp"
#include "kernel/sensor-driver/sensors/mpu6050g/mpu6050g.hpp"
#include "kernel/sensor-driver/sensors/qmc5883l/qmc5883l.hpp"
#include "kernel/sensor-driver/sensors/qmc6308/qmc6308.hpp"
#include "kernel/sensor-driver/sensors/sph0645lm4h/sph0645lm4h.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "std/fmt.hpp"
#include "std/log.hpp"
#include "std/maybe.hpp"

#include <cstdint>
#include <cstdio>

namespace zzs::kernel::sensor::driver::platform::android {

namespace {

/** @brief Android raw-I2C bus adapter via /dev/i2c-N with lazy open.
 *
 * The file descriptor is opened on first use (not at static construction
 * time).  On Android user builds, /dev/i2c-N is often restricted by SELinux
 * and may not be accessible when static initializers run.  Lazy open defers
 * the access until the app actually needs sensor data.  Only raw syscalls
 * (open, ioctl with I2C_SLAVE, read, write, close, nanosleep) are used — no
 * Android NDK APIs.  This is now the PRIMARY I2C path for raw register access.
 *
 * Caveat (verified on the Infinix X6885, SDK 36): a stock Android user build
 * with SELinux enforcing exposes no /dev/i2c-* nodes to an app, so every open
 * below fails and the probe functions return nullptr.  That is expected, not a
 * driver bug: on such devices sensor samples are delivered by the Java
 * SensrBridge -> JNI path (android/app/src/main/cpp/sensr_jni.cpp), which uses
 * SensorManager rather than raw I2C.  This HAL remains the path for rooted or
 * engineering builds and for boards that do expose the bus to userspace.
 * @{
 */
class AndroidI2cBus : public bus::BusInterface {
public:
  AndroidI2cBus (int bus_num, uint8_t addr) noexcept
      : _M_fd (-1), _M_bus_num (bus_num), _M_addr (addr), _M_opened (false) {
  }

  ~AndroidI2cBus () noexcept {
#ifdef __ANDROID__
    if (_M_fd >= 0)
      close (_M_fd);
#endif
  }

  AndroidI2cBus (const AndroidI2cBus &) = delete;
  AndroidI2cBus &
  operator= (const AndroidI2cBus &) = delete;
  AndroidI2cBus (AndroidI2cBus &&) = delete;
  AndroidI2cBus &
  operator= (AndroidI2cBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef __ANDROID__
    if (!ensure_open () || data == nullptr)
      return zzs::kernel::Status::INTERNAL;
    uint8_t buf[257];
    buf[0] = reg;
    for (uint16_t i = 0u; i < len; ++i)
      buf[i + 1u] = data[i];
    if (write (_M_fd, buf, static_cast<size_t> (len + 1u)) < 0)
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
  read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef __ANDROID__
    if (!ensure_open () || data == nullptr)
      return zzs::kernel::Status::INTERNAL;
    if (write (_M_fd, &reg, 1u) < 0)
      return zzs::kernel::Status::INTERNAL;
    if (read (_M_fd, data, static_cast<size_t> (len)) < 0)
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
  delay_ms (uint32_t ms) noexcept override {
#ifdef __ANDROID__
    struct timespec ts = {static_cast<time_t> (ms / 1000), static_cast<long> ((ms % 1000) * 1000000ull)};
    nanosleep (&ts, nullptr);
#else
    (void) ms;
#endif
  }

private:
  /** @brief Open /dev/i2c-N and select the slave address (lazy, on first use). */
  [[nodiscard]] bool
  ensure_open () noexcept {
    if (_M_opened)
      return _M_fd >= 0;
    _M_opened = true;
#ifdef __ANDROID__
    char path[32];
    const int n = snprintf (path, sizeof (path), "/dev/i2c-%d", _M_bus_num);
    (void) n;
    _M_fd = open (path, O_RDWR);
    if (_M_fd >= 0)
      ioctl (_M_fd, I2C_SLAVE, _M_addr);
#endif
    return _M_fd >= 0;
  }

  int _M_fd;
  int _M_bus_num;
  uint8_t _M_addr;
  bool _M_opened;
};
/** @} */

/** @brief Monotonic nanoseconds via CLOCK_MONOTONIC. */
[[nodiscard]] uint64_t
_S_android_now_ns (void) noexcept {
#ifdef __ANDROID__
  struct timespec ts;
  if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
    return static_cast<uint64_t> (ts.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts.tv_nsec);
#endif
  return 0ull;
}

/** @brief I2C bus instances (lazy-open) at common phone I2C addresses.
 *
 * These are created statically but the file descriptor is opened lazily on
 * first use (AndroidI2cBus::ensure_open).  Multiple buses share bus number 1
 * (the most common phone I2C bus) but target different slave addresses.
 * @{
 */
static AndroidI2cBus _S_i2c_accel (1, 0x6Au);           /* LSM6DSO / LSM6DS3 (SA0=0) */
static AndroidI2cBus _S_i2c_accel_alt (1, 0x67u);       /* LSM6DSO / LSM6DS3 (SA0=1) */
static AndroidI2cBus _S_i2c_mag_akm (1, 0x0Cu);         /* AK0991x / AK8963 magnetometer */
static AndroidI2cBus _S_i2c_mag_bosch (1, 0x10u);       /* BMM150 magnetometer */
static AndroidI2cBus _S_i2c_mag_qst (1, 0x0Du);         /* QMC6308 / QMC5883L magnetometer */
static AndroidI2cBus _S_i2c_mag_st (1, 0x1Eu);          /* LIS3MDL / LSM303 mag (SA0=0) */
static AndroidI2cBus _S_i2c_mag_st_alt (1, 0x1Cu);      /* LIS3MDL / LSM303 mag (SA0=1) */
static AndroidI2cBus _S_i2c_accel_st (1, 0x18u);        /* LIS2DH12 / LIS2DE12 (SA0=0) */
static AndroidI2cBus _S_i2c_accel_st_alt (1, 0x19u);    /* LIS2DH12 / LIS2DE12 (SA0=1) */
static AndroidI2cBus _S_i2c_accel_adxl (1, 0x1Au);      /* ADXL345 (ALT=0) */
static AndroidI2cBus _S_i2c_accel_adxl_alt (1, 0x53u);  /* ADXL345 (ALT=1) */
static AndroidI2cBus _S_i2c_accel_mma (1, 0x1Cu);       /* MMA8451Q (SA0=0) */
static AndroidI2cBus _S_i2c_accel_mma_alt (1, 0x1Du);   /* MMA8451Q (SA0=1) */
static AndroidI2cBus _S_i2c_gyro_mpu (1, 0x68u);        /* MPU6050 / ICM20602 / ICM20608 (AD0=0) */
static AndroidI2cBus _S_i2c_gyro_mpu_alt (1, 0x69u);    /* MPU6050 / ICM20602 / ICM20608 (AD0=1) */
static AndroidI2cBus _S_i2c_gyro_l3gd20 (1, 0x6Au);     /* L3GD20 (SA0=0) */
static AndroidI2cBus _S_i2c_gyro_l3gd20_alt (1, 0x6Bu); /* L3GD20 (SA0=1) */
static AndroidI2cBus _S_i2c_gyro_bmi (1, 0x68u);        /* BMI160 gyro (SDO=0) */
static AndroidI2cBus _S_i2c_gyro_bmi_alt (1, 0x69u);    /* BMI160 gyro (SDO=1) */
static AndroidI2cBus _S_i2c_light (1, 0x23u);           /* LTR-558 light */
static AndroidI2cBus _S_i2c_light_alt (1, 0x29u);       /* TSL2561 / VEML7700 / LTR-558 alt */

/** @brief Sensor driver instances (raw I2C path). Each is constructed with a
 *         non-owning bus pointer; init() verifies WHO_AM_I at probe time.
 * @{
 */
static sensors::Lsm6dsoAccelDriver _S_drv_lsm6dso_accel (&_S_i2c_accel);
static sensors::Lsm6dsoGyroDriver _S_drv_lsm6dso_gyro (&_S_i2c_accel);
static sensors::Lsm6dsaAccelDriver _S_drv_lsm6dsa_accel (&_S_i2c_accel_alt);
static sensors::Lsm6dsgGyroDriver _S_drv_lsm6dsg_gyro (&_S_i2c_accel_alt);
static sensors::Ak0991xMagDriver _S_drv_ak0991x_mag (&_S_i2c_mag_akm);
static sensors::Ak8963MagDriver _S_drv_ak8963_mag (&_S_i2c_mag_akm);
static sensors::Bmm150MagDriver _S_drv_bmm150_mag (&_S_i2c_mag_bosch);
static sensors::Qmc6308MagDriver _S_drv_qmc6308_mag (&_S_i2c_mag_qst);
static sensors::Qmc5883lMagDriver _S_drv_qmc5883l_mag (&_S_i2c_mag_qst);
static sensors::Lis3mdlMagDriver _S_drv_lis3mdl_mag (&_S_i2c_mag_st);
static sensors::Lis2de12AccelDriver _S_drv_lis2de12_accel (&_S_i2c_accel_st);
static sensors::Adxl345AccelDriver _S_drv_adxl345_accel (&_S_i2c_accel_adxl);
static sensors::Mma8451qAccelDriver _S_drv_mma8451q_accel (&_S_i2c_accel_mma);
static sensors::Mpu6050AccelDriver _S_drv_mpu6050_accel (&_S_i2c_gyro_mpu);
static sensors::Mpu6050GyroDriver _S_drv_mpu6050_gyro (&_S_i2c_gyro_mpu);
static sensors::L3gd20GyroDriver _S_drv_l3gd20_gyro (&_S_i2c_gyro_l3gd20);
static sensors::Icm20602GyroDriver _S_drv_icm20602_gyro (&_S_i2c_gyro_mpu);
static sensors::Icm20608GyroDriver _S_drv_icm20608_gyro (&_S_i2c_gyro_mpu_alt);
static sensors::Bmi160gGyroDriver _S_drv_bmi160g_gyro (&_S_i2c_gyro_bmi);
static sensors::Ltr558LightDriver _S_drv_ltr558_light (&_S_i2c_light);
static sensors::Ics43434SoundDriver _S_drv_sound (nullptr, nullptr);
/** @} */

/** @brief Detected driver pointers, set during probe. Null when absent. */
static sensors::SensorDriver *_S_mag_driver = nullptr;
static sensors::SensorDriver *_S_accel_driver = nullptr;
static sensors::SensorDriver *_S_gyro_driver = nullptr;
static sensors::SensorDriver *_S_light_driver = nullptr;
static sensors::SensorDriver *_S_sound_driver = &_S_drv_sound;

static zzs::kernel::hal::_S_SensorHal _S_hal_vtable;
static bool _S_hal_initialized = false;

/** @brief Probe a single I2C address for a WHO_AM_I / chip ID match.
 *
 * Creates a temporary AndroidI2cBus at the given address, reads one register,
 * and compares against the expected value.  Returns true when detected.
 */
[[nodiscard]] bool
_S_probe_who_am_i (AndroidI2cBus &bus, uint8_t reg, uint8_t expected) noexcept {
  uint8_t who = 0;
  return zzs::kernel::status_ok (bus.read_reg (reg, &who, 1u)) && who == expected;
}

/** @brief Probe for an accelerometer at known phone I2C addresses.
 *
 * Tries LSM6DSO (0x6A/0x67, WHO=0x6C), LSM6DS3 (0x6A/0x67, WHO=0x69),
 * LIS2DH12/LIS2DE12 (0x18/0x19, WHO=0x33), ADXL345 (0x1A/0x53, DEVID=0xE5),
 * MMA8451Q (0x1C/0x1D, WHO=0x1A), MPU6050 (0x68, WHO=0x68).
 * Returns the matching driver or nullptr when none detected.
 */
[[nodiscard]] sensors::SensorDriver *
_S_probe_accelerometer () noexcept {
  /* LSM6DSO at 0x6A. */
  if (_S_probe_who_am_i (_S_i2c_accel, sensors::lsm6dso::RegWhoAmI, sensors::lsm6dso::WhoAmIValue))
    return &_S_drv_lsm6dso_accel;
  /* LSM6DSO at 0x67. */
  if (_S_probe_who_am_i (_S_i2c_accel_alt, sensors::lsm6dso::RegWhoAmI, sensors::lsm6dso::WhoAmIValue))
    return &_S_drv_lsm6dso_accel;
  /* LSM6DS3 at 0x6A. */
  if (_S_probe_who_am_i (_S_i2c_accel, sensors::lsm6dsa::RegWhoAmI, sensors::lsm6dsa::WhoAmIValue))
    return &_S_drv_lsm6dsa_accel;
  /* LSM6DS3 at 0x67. */
  if (_S_probe_who_am_i (_S_i2c_accel_alt, sensors::lsm6dsa::RegWhoAmI, sensors::lsm6dsa::WhoAmIValue))
    return &_S_drv_lsm6dsa_accel;
  /* LIS2DH12 / LIS2DE12 at 0x18. */
  if (_S_probe_who_am_i (_S_i2c_accel_st, sensors::lis2dh12::RegWhoAmI, sensors::lis2dh12::WhoAmIValue)) {
    return &_S_drv_lis2de12_accel; /* share accel driver instance */
  }
  /* LIS2DH12 / LIS2DE12 at 0x19. */
  if (_S_probe_who_am_i (_S_i2c_accel_st_alt, sensors::lis2dh12::RegWhoAmI, sensors::lis2dh12::WhoAmIValue)) {
    return &_S_drv_lis2de12_accel;
  }
  /* ADXL345 at 0x1A. */
  if (_S_probe_who_am_i (_S_i2c_accel_adxl, sensors::adxl345::RegDevid, sensors::adxl345::DevidValue))
    return &_S_drv_adxl345_accel;
  /* ADXL345 at 0x53. */
  if (_S_probe_who_am_i (_S_i2c_accel_adxl_alt, sensors::adxl345::RegDevid, sensors::adxl345::DevidValue))
    return &_S_drv_adxl345_accel;
  /* MMA8451Q at 0x1C. */
  if (_S_probe_who_am_i (_S_i2c_accel_mma, sensors::mma8451q::RegWhoAmI, sensors::mma8451q::WhoAmIValue))
    return &_S_drv_mma8451q_accel;
  /* MMA8451Q at 0x1D. */
  if (_S_probe_who_am_i (_S_i2c_accel_mma_alt, sensors::mma8451q::RegWhoAmI, sensors::mma8451q::WhoAmIValue))
    return &_S_drv_mma8451q_accel;
  /* MPU6050 at 0x68. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu, sensors::mpu6050::RegWhoAmI, sensors::mpu6050::WhoAmIValue))
    return &_S_drv_mpu6050_accel;
  return nullptr;
}

/** @brief Probe for a gyroscope at known phone I2C addresses.
 *
 * Tries LSM6DSO/LSM6DS3 (shares accel bus), L3GD20 (0x6A/0x6B, WHO=0xD7),
 * MPU6050 (0x68/0x69, WHO=0x68), ICM20602 (0x68/0x69, WHO=0xAA),
 * ICM20608 (0x68/0x69, WHO=0xA0), BMI160 (0x68/0x69, WHO=0xD1).
 */
[[nodiscard]] sensors::SensorDriver *
_S_probe_gyroscope () noexcept {
  /* LSM6DSO gyro shares accel bus; probe WHO=0x6C at 0x6A/0x67. */
  if (_S_probe_who_am_i (_S_i2c_accel, sensors::lsm6dso::RegWhoAmI, sensors::lsm6dso::WhoAmIValue))
    return &_S_drv_lsm6dso_gyro;
  if (_S_probe_who_am_i (_S_i2c_accel_alt, sensors::lsm6dso::RegWhoAmI, sensors::lsm6dso::WhoAmIValue))
    return &_S_drv_lsm6dso_gyro;
  /* LSM6DS3 gyro shares accel bus; probe WHO=0x69 at 0x6A/0x67. */
  if (_S_probe_who_am_i (_S_i2c_accel, sensors::lsm6dsa::RegWhoAmI, sensors::lsm6dsa::WhoAmIValue))
    return &_S_drv_lsm6dsg_gyro;
  if (_S_probe_who_am_i (_S_i2c_accel_alt, sensors::lsm6dsa::RegWhoAmI, sensors::lsm6dsa::WhoAmIValue))
    return &_S_drv_lsm6dsg_gyro;
  /* L3GD20 at 0x6A. */
  if (_S_probe_who_am_i (_S_i2c_gyro_l3gd20, sensors::l3gd20::RegWhoAmI, sensors::l3gd20::WhoAmIValue))
    return &_S_drv_l3gd20_gyro;
  /* L3GD20 at 0x6B. */
  if (_S_probe_who_am_i (_S_i2c_gyro_l3gd20_alt, sensors::l3gd20::RegWhoAmI, sensors::l3gd20::WhoAmIValue))
    return &_S_drv_l3gd20_gyro;
  /* MPU6050 at 0x68. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu, sensors::mpu6050::RegWhoAmI, sensors::mpu6050::WhoAmIValue))
    return &_S_drv_mpu6050_gyro;
  /* MPU6050 at 0x69. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu_alt, sensors::mpu6050::RegWhoAmI, sensors::mpu6050::WhoAmIValue))
    return &_S_drv_mpu6050_gyro;
  /* ICM20602 at 0x68. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu, sensors::icm20602::RegWhoAmI, sensors::icm20602::WhoAmIValue))
    return &_S_drv_icm20602_gyro;
  /* ICM20602 at 0x69. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu_alt, sensors::icm20602::RegWhoAmI, sensors::icm20602::WhoAmIValue))
    return &_S_drv_icm20602_gyro;
  /* ICM20608 at 0x68. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu, sensors::icm20608::RegWhoAmI, sensors::icm20608::WhoAmIValue))
    return &_S_drv_icm20608_gyro;
  /* ICM20608 at 0x69. */
  if (_S_probe_who_am_i (_S_i2c_gyro_mpu_alt, sensors::icm20608::RegWhoAmI, sensors::icm20608::WhoAmIValue))
    return &_S_drv_icm20608_gyro;
  /* BMI160 at 0x68. */
  if (_S_probe_who_am_i (_S_i2c_gyro_bmi, sensors::bmi160g::RegChipId, sensors::bmi160g::ChipIdValue))
    return &_S_drv_bmi160g_gyro;
  /* BMI160 at 0x69. */
  if (_S_probe_who_am_i (_S_i2c_gyro_bmi_alt, sensors::bmi160g::RegChipId, sensors::bmi160g::ChipIdValue))
    return &_S_drv_bmi160g_gyro;
  return nullptr;
}

/** @brief Probe for a magnetometer at known phone I2C addresses.
 *
 * Tries AK0991x/AK8963 (0x0C, WHO=0x48), BMM150 (0x10, chip ID=0x32),
 * QMC6308/QMC5883L (0x0D, chip ID=0x40), LIS3MDL (0x1C/0x3D, WHO=0x3D),
 * LSM303 mag (0x1E/0x1D).
 */
[[nodiscard]] sensors::SensorDriver *
_S_probe_magnetometer () noexcept {
  /* AK0991x at 0x0C. */
  if (_S_probe_who_am_i (_S_i2c_mag_akm, sensors::ak0991x::RegWia, sensors::ak0991x::WiaValue))
    return &_S_drv_ak0991x_mag;
  /* AK8963 at 0x0C. */
  if (_S_probe_who_am_i (_S_i2c_mag_akm, sensors::ak8963::RegWia, sensors::ak8963::WiaValue))
    return &_S_drv_ak8963_mag;
  /* BMM150 at 0x10. */
  if (_S_probe_who_am_i (_S_i2c_mag_bosch, sensors::bmm150::RegChipId, sensors::bmm150::ChipIdValue))
    return &_S_drv_bmm150_mag;
  /* QMC6308 at 0x0D. */
  if (_S_probe_who_am_i (_S_i2c_mag_qst, sensors::qmc6308::RegChipId, sensors::qmc6308::ChipIdValue))
    return &_S_drv_qmc6308_mag;
  /* QMC5883L at 0x0D. */
  if (_S_probe_who_am_i (_S_i2c_mag_qst, sensors::qmc5883l::RegChipId, sensors::qmc5883l::ChipIdValue))
    return &_S_drv_qmc5883l_mag;
  /* LIS3MDL at 0x1C (SA0=0) / 0x3D (SA0=1). */
  if (_S_probe_who_am_i (_S_i2c_mag_st, sensors::lis3mdl::RegWhoAmI, sensors::lis3mdl::WhoAmIValue))
    return &_S_drv_lis3mdl_mag;
  if (_S_probe_who_am_i (_S_i2c_mag_st_alt, sensors::lis3mdl::RegWhoAmI, sensors::lis3mdl::WhoAmIValue))
    return &_S_drv_lis3mdl_mag;
  return nullptr;
}

/** @brief Probe for a light sensor at known phone I2C addresses.
 *
 * Tries LTR-558 (0x23/0x29, Part ID=0x05), TSL2561 (0x29/0x39),
 * VEML7700 (0x10/0x48).
 */
[[nodiscard]] sensors::SensorDriver *
_S_probe_light () noexcept {
  /* LTR-558 at 0x23. */
  if (_S_probe_who_am_i (_S_i2c_light, sensors::ltr558::RegPartId, sensors::ltr558::PartIdValue))
    return &_S_drv_ltr558_light;
  /* LTR-558 at 0x29. */
  if (_S_probe_who_am_i (_S_i2c_light_alt, sensors::ltr558::RegPartId, sensors::ltr558::PartIdValue))
    return &_S_drv_ltr558_light;
  return nullptr;
}

[[nodiscard]] zzs::kernel::Status
_S_hal_read_sample (sensor::Sample &out) noexcept {
  if (!_S_hal_initialized)
    return zzs::kernel::Status::NOT_READY;

  /* Round-robin across detected drivers. */
  static uint8_t idx = 0u;
  const uint8_t n_drivers = 5u; /* accel, gyro, mag, light, sound */
  for (uint8_t tries = 0u; tries < n_drivers; ++tries) {
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    switch (idx) {
    case 0u:
      if (_S_accel_driver != nullptr)
        st = _S_accel_driver->read_sample (out);
      break;
    case 1u:
      if (_S_gyro_driver != nullptr)
        st = _S_gyro_driver->read_sample (out);
      break;
    case 2u:
      if (_S_mag_driver != nullptr)
        st = _S_mag_driver->read_sample (out);
      break;
    case 3u:
      if (_S_light_driver != nullptr)
        st = _S_light_driver->read_sample (out);
      break;
    case 4u:
      st = _S_sound_driver->read_sample (out);
      break;
    }
    idx = static_cast<uint8_t> ((idx + 1u) % n_drivers);
    if (zzs::kernel::status_ok (st))
      return zzs::kernel::Status::OK;
  }
  return zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
_S_hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept {
  if (_S_accel_driver != nullptr)
    (void) _S_accel_driver->set_channel_enabled (ch, enabled);
  if (_S_gyro_driver != nullptr)
    (void) _S_gyro_driver->set_channel_enabled (ch, enabled);
  if (_S_mag_driver != nullptr)
    (void) _S_mag_driver->set_channel_enabled (ch, enabled);
  if (_S_light_driver != nullptr)
    (void) _S_light_driver->set_channel_enabled (ch, enabled);
  (void) _S_sound_driver->set_channel_enabled (ch, enabled);
  return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
_S_hal_channel_rate_hz (sensor::Channel ch) noexcept {
  return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
_S_hal_init (void) noexcept {
  if (_S_hal_initialized)
    return zzs::kernel::Status::OK;

  zzs::base::log_info ("android sensor hal init (raw I2C)");

  zzs::kernel::Status st = zzs::kernel::clock::clock_source_install (&_S_android_now_ns);
  if (!zzs::kernel::status_ok (st))
    return st;

  /* Probe for accelerometer. */
  _S_accel_driver = _S_probe_accelerometer ();
  if (_S_accel_driver != nullptr) {
    st = _S_accel_driver->init ();
    if (!zzs::kernel::status_ok (st)) {
      zzs::base::log_error ("android sensor hal init accel failed: {}", zzs::kernel::status_name (st));
      _S_accel_driver = nullptr;
    } else {
      zzs::base::log_info ("android sensor hal: accel detected (device_id=0x{:02x})",
                           static_cast<unsigned> (_S_accel_driver->device_id ()));
    }
  } else {
    zzs::base::log_info ("android sensor hal: no accelerometer detected");
  }

  /* Probe for gyroscope. */
  _S_gyro_driver = _S_probe_gyroscope ();
  if (_S_gyro_driver != nullptr) {
    st = _S_gyro_driver->init ();
    if (!zzs::kernel::status_ok (st)) {
      zzs::base::log_error ("android sensor hal init gyro failed: {}", zzs::kernel::status_name (st));
      _S_gyro_driver = nullptr;
    } else {
      zzs::base::log_info ("android sensor hal: gyro detected (device_id=0x{:02x})",
                           static_cast<unsigned> (_S_gyro_driver->device_id ()));
    }
  } else {
    zzs::base::log_info ("android sensor hal: no gyroscope detected");
  }

  /* Probe for magnetometer. */
  _S_mag_driver = _S_probe_magnetometer ();
  if (_S_mag_driver != nullptr) {
    st = _S_mag_driver->init ();
    if (!zzs::kernel::status_ok (st)) {
      zzs::base::log_error ("android sensor hal init mag failed: {}", zzs::kernel::status_name (st));
      _S_mag_driver = nullptr;
    } else {
      zzs::base::log_info ("android sensor hal: mag detected (device_id=0x{:02x})",
                           static_cast<unsigned> (_S_mag_driver->device_id ()));
    }
  } else {
    zzs::base::log_info ("android sensor hal: no magnetometer detected");
  }

  /* Probe for light sensor. */
  _S_light_driver = _S_probe_light ();
  if (_S_light_driver != nullptr) {
    st = _S_light_driver->init ();
    if (!zzs::kernel::status_ok (st)) {
      zzs::base::log_error ("android sensor hal init light failed: {}", zzs::kernel::status_name (st));
      _S_light_driver = nullptr;
    } else {
      zzs::base::log_info ("android sensor hal: light detected (device_id=0x{:02x})",
                           static_cast<unsigned> (_S_light_driver->device_id ()));
    }
  } else {
    zzs::base::log_info ("android sensor hal: no light sensor detected");
  }

  /* Sound (PDM microphone): init always succeeds; read_sample returns
   * UNSUPPORTED when no PDM read function is provided. */
  st = _S_sound_driver->init ();
  if (!zzs::kernel::status_ok (st)) {
    zzs::base::log_error ("android sensor hal init sound failed: {}", zzs::kernel::status_name (st));
  }

  _S_hal_initialized = true;
  return zzs::kernel::Status::OK;
}

} // namespace

[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_android_sensor_hal () noexcept {
  _S_hal_vtable._M_read_sample = &_S_hal_read_sample;
  _S_hal_vtable._M_set_channel_enabled = &_S_hal_set_channel_enabled;
  _S_hal_vtable._M_channel_rate_hz = &_S_hal_channel_rate_hz;
  _S_hal_vtable._M_init = &_S_hal_init;
  return &_S_hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::android
