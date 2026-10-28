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

#include "mllee/mllee.hpp"

#include <cstdint>

/**
 * @brief Stack-usage probe.
 *
 * The Core object is owned by the caller so its bytes are not charged to the
 * core call stack.  Compile this file with -fstack-usage and inspect the .su
 * output for mllee_worst_case and any outlined mllee callees.
 */
extern "C" int
mllee_worst_case (zzs::mllee::Core &core, const uint8_t *model, const uint32_t model_len,
                  const std::int8_t *input, const uint32_t input_len)
{
    const auto result = core.run (model, model_len, input, input_len);
    return result.has_value () ? 1 : 0;
}
