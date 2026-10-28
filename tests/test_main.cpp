/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * @file test_main.cpp
 * @brief Catch2 entry point that initializes the std runtime.
 *
 * The std heap allocator must be initialized before any sensr backend can be
 * created (all backends allocate from it). Logging is likewise initialized, but
 * kept quiet to avoid drowning the test output.
 */

#include <catch2/catch_session.hpp>

extern "C"
{
#include "std/heap_allocator.h"
#include "std/log.h"
}

int
main (int argc, char **argv)
{
        (void) ZZS_std_heap_init (32u * 1024u * 1024u);
        (void) ZZS_std_log_init ();
        ZZS_std_log_set_min_level (ZZS_LOG_LEVEL_WARN);

        int result = Catch::Session ().run (argc, argv);

        ZZS_std_log_shutdown ();
        ZZS_std_heap_shutdown ();
        return result;
}
