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

#include <cstdio>

namespace
{
template <typename T>
void
print_size (const char *const name)
{
    std::printf ("sizeof(%s)=%zu alignof(%s)=%zu\n", name, sizeof (T), name, alignof (T));
}
} // namespace

int
main ()
{
    print_size<zzs::mllee::Core> ("Core");
    print_size<zzs::mllee::PacklowState> ("PacklowState");
    print_size<zzs::mllee::Layer> ("Layer");
    print_size<zzs::mllee::Header> ("Header");
    print_size<zzs::mllee::Core::Output> ("Output");
    std::printf ("arena=%u\n", static_cast<unsigned int> (zzs::mllee::_S_arena_bytes));
    std::printf ("fixed=%zu\n", sizeof (zzs::mllee::Core) - zzs::mllee::_S_arena_bytes);
    std::printf ("budget=%u core=%zu chain=%u\n",
                 static_cast<unsigned int> (zzs::mllee::_S_ram_budget_bytes),
                 sizeof (zzs::mllee::Core),
                 static_cast<unsigned int> (zzs::mllee::_S_stack_budget_bytes));

    constexpr bool _S_32bit = sizeof (void *) == 4;
    constexpr std::size_t _S_core_budget
        = _S_32bit ? zzs::mllee::_S_core_budget_bytes
                   : zzs::mllee::_S_core_budget_host_bytes;
    constexpr std::size_t _S_pk_budget = _S_32bit ? 124u : 136u;
    constexpr std::size_t _S_ws_budget = _S_32bit ? 192u : 208u;

    // Exact 32-bit (Cortex-M4) layout; this is what check_mllee_ram.sh gates.
    static_assert (!_S_32bit || sizeof (zzs::mllee::Core) == 640u);
    static_assert (!_S_32bit || sizeof (zzs::mllee::Core) + zzs::mllee::_S_stack_budget_bytes
                                          == zzs::mllee::_S_ram_budget_bytes);
    static_assert (!_S_32bit || sizeof (zzs::mllee::Workspace) == 192u);
    static_assert (!_S_32bit || sizeof (zzs::mllee::PacklowState) == 124u);
    static_assert (!_S_32bit || alignof (zzs::mllee::Core) == 4u);

    bool ok = true;
    if (sizeof (zzs::mllee::Core) > _S_core_budget)
        {
            std::printf ("FAIL: Core exceeds the %zu-byte object budget\n", _S_core_budget);
            ok = false;
        }
    if (_S_32bit && alignof (zzs::mllee::Core) > 4)
        {
            std::printf ("FAIL: Core alignment exceeds 4 bytes\n");
            ok = false;
        }
    if (sizeof (zzs::mllee::PacklowState) > _S_pk_budget)
        {
            std::printf ("FAIL: PacklowState exceeds %zu bytes\n", _S_pk_budget);
            ok = false;
        }
    if (sizeof (zzs::mllee::Workspace) > _S_ws_budget)
        {
            std::printf ("FAIL: Workspace exceeds %zu bytes\n", _S_ws_budget);
            ok = false;
        }
    if (sizeof (zzs::mllee::Layer) != zzs::mllee::_S_layer_desc_size)
        {
            std::printf ("FAIL: Layer descriptor size does not match the blob format\n");
            ok = false;
        }
    return ok ? 0 : 1;
}
