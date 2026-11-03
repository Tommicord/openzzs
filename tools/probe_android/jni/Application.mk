# Copyright(c) 2026 Tommicord
# MIT License - see LICENSE at repo root

APP_ABI := armeabi-v7a arm64-v8a
APP_PLATFORM := android-24
APP_STL := c++_shared
APP_CPPFLAGS := -std=c++23 -fno-exceptions -fno-rtti -DZZS_KERNEL_PLATFORM_ANDROID -D__ANDROID__
APP_OPTIM := release
NDK_TOOLCHAIN_VERSION := clang
