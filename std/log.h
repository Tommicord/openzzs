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

#ifndef ZZS_STD_LOG_H
#define ZZS_STD_LOG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Log severity levels. */
typedef enum ZZS_std_log_level {
    ZZS_LOG_LEVEL_ERROR = 1,
    ZZS_LOG_LEVEL_WARN = 2,
    ZZS_LOG_LEVEL_INFO = 3,
    ZZS_LOG_LEVEL_DEBUG = 4,
} ZZS_std_log_level_t;

/** @brief Initialise the logging subsystem. */
static inline void
ZZS_std_log_init (void) noexcept {
}

/** @brief Set the minimum log level. */
static inline void
ZZS_std_log_set_min_level (ZZS_std_log_level_t level) noexcept {
    (void) level;
}

/** @brief Shut down the logging subsystem. */
static inline void
ZZS_std_log_shutdown (void) noexcept {
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* ZZS_STD_LOG_H */
