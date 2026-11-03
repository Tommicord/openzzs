# Copyright(c) 2026 Tommicord
# MIT License - see LICENSE at repo root

LOCAL_PATH := $(call my-dir)

OPENZZS_ROOT := $(LOCAL_PATH)/../../..

include $(CLEAR_VARS)

LOCAL_MODULE := openzzs_probe

LOCAL_CFLAGS := \
  -DZZS_KERNEL_PLATFORM_ANDROID \
  -D__ANDROID__ \
  -O2 \
  -Wall \
  -Wextra \
  -fno-exceptions \
  -fno-rtti \
  -std=c++23

LOCAL_C_INCLUDES := \
  $(OPENZZS_ROOT) \
  $(OPENZZS_ROOT)/std \
  $(OPENZZS_ROOT)/kernel \
  $(OPENZZS_ROOT)/kernel/sensor-driver \
  $(OPENZZS_ROOT)/kernel/sensor-driver/platforms \
  $(OPENZZS_ROOT)/mllee \
  $(OPENZZS_ROOT)/tools/probe_android/jni

LOCAL_SRC_FILES := \
  probe_main.cpp \
  $(OPENZZS_ROOT)/kernel/sensor-driver/platforms/android/platform_hal.cpp

LOCAL_LDLIBS := -llog -landroid

include $(BUILD_SHARED_LIBRARY)
