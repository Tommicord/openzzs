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

#include "platform_hal.hpp"

#include <cstdint>

#if defined(STM32F4)
#if defined(__has_include)
#if __has_include("stm32f4xx_hal.h")
#include "stm32f4xx_hal.h"
#define ZZS_GNSS_STM32_USE_HAL 1
#endif
#endif
#endif

#ifndef ZZS_GNSS_STM32_USE_HAL
/* ------------------------------------------------------------------
 * Portable direct-register fallback for STM32F4 USART.
 *
 * Defines the STM32F4 register blocks manually so this translation unit
 * compiles without the vendor SDK.  When built on a real STM32F4 target
 * (with -DSTM32F4), these map to the actual hardware addresses and the
 * UART functions are functional.  On other targets they return
 * UNSUPPORTED (no hardware present) but the code still compiles.
 * ------------------------------------------------------------------ */
namespace zzs::kernel::device::driver::gnss::platform::stm32::f4::detail {

// ------------------------------------------------------------------
// RCC (Reset and Clock Control) - base 0x40023800
// ------------------------------------------------------------------
struct RccRegs {
  volatile uint32_t _M_cr;       // 0x00
  volatile uint32_t _M_pllcfgr;  // 0x04
  volatile uint32_t _M_cfgr;     // 0x08
  volatile uint32_t _M_cir;      // 0x0C
  volatile uint32_t _M_ahb1rstr; // 0x10
  volatile uint32_t _M_ahb2rstr; // 0x14
  volatile uint32_t _M_ahb3rstr; // 0x18
  uint32_t _M_reserved0;         // 0x1C
  volatile uint32_t _M_apb1rstr; // 0x20
  volatile uint32_t _M_apb2rstr; // 0x24
  uint32_t _M_reserved1[2];      // 0x28, 0x2C
  volatile uint32_t _M_ahb1enr;  // 0x30
  volatile uint32_t _M_ahb2enr;  // 0x34
  volatile uint32_t _M_ahb3enr;  // 0x38
  uint32_t _M_reserved2;         // 0x3C
  volatile uint32_t _M_apb1enr;  // 0x40
  volatile uint32_t _M_apb2enr;  // 0x44
};

// ------------------------------------------------------------------
// GPIO - base 0x40020000 (GPIOA)
// ------------------------------------------------------------------
struct GpioRegs {
  volatile uint32_t _M_moder;   // 0x00
  volatile uint32_t _M_otyper;  // 0x04
  volatile uint32_t _M_ospeedr; // 0x08
  volatile uint32_t _M_pupdr;   // 0x0C
  volatile uint32_t _M_idr;     // 0x10
  volatile uint32_t _M_odr;     // 0x14
  volatile uint32_t _M_bsrr;    // 0x18
  volatile uint32_t _M_lckr;    // 0x1C
  volatile uint32_t _M_afr[2];  // 0x20, 0x24
};

// ------------------------------------------------------------------
// USART - base 0x40004400 (USART2)
// ------------------------------------------------------------------
struct UsartRegs {
  volatile uint32_t _M_sr;  // 0x00 Status register
  volatile uint32_t _M_dr;  // 0x04 Data register
  volatile uint32_t _M_brr; // 0x08 Baud rate register
  volatile uint32_t _M_cr1; // 0x0C Control register 1
  volatile uint32_t _M_cr2; // 0x10 Control register 2
  volatile uint32_t _M_cr3; // 0x14 Control register 3
  volatile uint32_t _M_gpr; // 0x18 Guard time/prescaler
};

// SR bits
inline constexpr uint32_t USART_SR_TXE = 0x0080u;  // Transmit data register empty
inline constexpr uint32_t USART_SR_RXNE = 0x0020u; // Read data register not empty
// CR1 bits
inline constexpr uint32_t USART_CR1_UE = 0x0001u; // USART enable
inline constexpr uint32_t USART_CR1_TE = 0x0008u; // Transmitter enable
inline constexpr uint32_t USART_CR1_RE = 0x0004u; // Receiver enable

inline constexpr uintptr_t RCC_BASE = 0x40023800u;
inline constexpr uintptr_t GPIOA_BASE = 0x40020000u;
inline constexpr uintptr_t USART2_BASE = 0x40004400u;

inline RccRegs &
rcc () noexcept {
  return *reinterpret_cast<RccRegs *> (RCC_BASE);
}
inline GpioRegs &
gpioa () noexcept {
  return *reinterpret_cast<GpioRegs *> (GPIOA_BASE);
}
inline UsartRegs &
usart2 () noexcept {
  return *reinterpret_cast<UsartRegs *> (USART2_BASE);
}

} // namespace zzs::kernel::device::driver::gnss::platform::stm32::f4::detail
#endif // !ZZS_GNSS_STM32_USE_HAL

namespace zzs::kernel::device::driver::gnss::platform::stm32::f4 {

namespace {

/** @brief STM32F4 UART byte-stream bus adapter (direct registers or HAL). */
class Stm32f4GnssBus {
public:
  Stm32f4GnssBus () noexcept = default;
  ~Stm32f4GnssBus () noexcept = default;

  Stm32f4GnssBus (const Stm32f4GnssBus &) = delete;
  Stm32f4GnssBus &
  operator= (const Stm32f4GnssBus &) = delete;
  Stm32f4GnssBus (Stm32f4GnssBus &&) = delete;
  Stm32f4GnssBus &
  operator= (Stm32f4GnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(ZZS_GNSS_STM32_USE_HAL)
    /* STM32Cube HAL path.  The caller must have already initialized HAL. */
    (void) 0;
    return zzs::kernel::Status::OK;
#elif defined(STM32F4)
    /* Direct-register path: enable GPIOA + USART2 clocks, configure PA2/TX
     * and PA3/RX as alternate function AF7, set 9600 baud 8N1, enable UART. */
    using namespace detail;
    rcc ()._M_ahb1enr |= 0x00000001u; // GPIOAEN
    rcc ()._M_apb1enr |= 0x00020000u; // USART2EN

    /* PA2 = USART2_TX (AF7), PA3 = USART2_RX (AF7): mode = alternate func. */
    gpioa ()._M_moder &= ~(0x000000F0u);  // clear PA2/PA3 mode
    gpioa ()._M_moder |= 0x000000A0u;     // PA2/PA3 = alternate function
    gpioa ()._M_afr[0] &= ~(0x0000FF00u); // clear PA2/PA3 AF
    gpioa ()._M_afr[0] |= 0x00007700u;    // PA2/PA3 = AF7 (USART2)

    /* Baud: USARTDIV = fCK / baud. With fCK = 42 MHz (APB1), 9600 baud:
     * USARTDIV = 42000000 / 9600 = 4375. BRR = 4375 = 0x1117 */
    usart2 ()._M_brr = 0x1117u;
    usart2 ()._M_cr1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
    return zzs::kernel::Status::OK;
#else
    /* Not an STM32F4 target: no UART hardware available. */
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  close () noexcept {
#if defined(ZZS_GNSS_STM32_USE_HAL)
    (void) 0;
    return zzs::kernel::Status::OK;
#elif defined(STM32F4)
    using namespace detail;
    usart2 ()._M_cr1 &= ~USART_CR1_UE;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    if (buf == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(ZZS_GNSS_STM32_USE_HAL)
    (void) buf;
    (void) max_bytes;
    *filled = 0u;
    return zzs::kernel::Status::STALE;
#elif defined(STM32F4)
    using namespace detail;
    uint16_t n = 0u;
    while (n < max_bytes) {
      if ((usart2 ()._M_sr & USART_SR_RXNE) == 0u)
        break;
      buf[n++] = static_cast<uint8_t> (usart2 ()._M_dr);
    }
    *filled = n;
    return n > 0u ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
#else
    (void) buf;
    (void) max_bytes;
    *filled = 0u;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(ZZS_GNSS_STM32_USE_HAL)
    (void) buf;
    (void) len;
    return zzs::kernel::Status::OK;
#elif defined(STM32F4)
    using namespace detail;
    for (uint16_t i = 0u; i < len; ++i) {
      while ((usart2 ()._M_sr & USART_SR_TXE) == 0u) {
      } // wait for TX empty
      usart2 ()._M_dr = buf[i];
    }
    return zzs::kernel::Status::OK;
#else
    (void) buf;
    (void) len;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(STM32F4)
    /* Simple busy-wait delay (approximate, core-clock dependent). */
    volatile uint32_t count = ms * 8000u;
    while (count-- > 0u) {
      __asm__ volatile ("nop");
    }
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(STM32F4)
    using namespace detail;
    /* Discard any pending RX data by reading DR while RXNE is set. */
    while ((usart2 ()._M_sr & USART_SR_RXNE) != 0u) {
      (void) usart2 ()._M_dr;
    }
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] bool
  is_open () const noexcept {
#if defined(STM32F4)
    using namespace detail;
    return (usart2 ()._M_cr1 & USART_CR1_UE) != 0u;
#else
    return false;
#endif
  }
};

/** @brief Shared bus instance (function-local static). */
Stm32f4GnssBus &
bus_ref () noexcept {
  static Stm32f4GnssBus _S_bus;
  return _S_bus;
}

[[nodiscard]] zzs::kernel::Status
_S_open (void) noexcept {
  return bus_ref ().open ();
}

[[nodiscard]] zzs::kernel::Status
_S_close (void) noexcept {
  return bus_ref ().close ();
}

[[nodiscard]] zzs::kernel::Status
_S_read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
  return bus_ref ().read_bytes (buf, max_bytes, filled);
}

[[nodiscard]] zzs::kernel::Status
_S_write_bytes (const uint8_t *buf, uint16_t len) noexcept {
  return bus_ref ().write_bytes (buf, len);
}

void
_S_delay_ms (uint32_t ms) noexcept {
  bus_ref ().delay_ms (ms);
}

[[nodiscard]] zzs::kernel::Status
_S_flush_rx (void) noexcept {
  return bus_ref ().flush_rx ();
}

[[nodiscard]] bool
_S_is_open (void) noexcept {
  return bus_ref ().is_open ();
}

/** @brief The single STM32F4 GNSS HAL vtable (const, .rodata). */
static const GnssPlatformHal _S_hal_vtable = {
    &_S_open, &_S_close, &_S_read_bytes, &_S_write_bytes, &_S_delay_ms, &_S_flush_rx, &_S_is_open,
};

} // namespace

[[nodiscard]] const GnssPlatformHal *
create_stm32f4_gnss_hal () noexcept {
  (void) bus_ref (); // ensure construction
  return &_S_hal_vtable;
}

} // namespace zzs::kernel::device::driver::gnss::platform::stm32::f4
