# Openzzs Sensr — Android host

Minimal Android application whose only purpose is to stream **real hardware
sensor data** into the portable C backend in [`../sensr`](../sensr). It does not
process, store or upload samples: `MainActivity` enables the sensors that exist
on the device, starts the native sampler, polls the event queue and prints every
event to an on-screen log.

## What gets exercised

| Layer | File |
| --- | --- |
| Kotlin entry point | `app/src/main/java/com/openzzs` |
| Java façade | `app/src/main/java/com/openzzs` |
| JNI bridge | `app/src/main/cpp/sensr_jni.c` |
| Native build | `app/src/main/cpp/CMakeLists.txt` |
| C backend | `../sensr/sensr_android.c` |

The module compiles the `sensr` and selected `std` sources directly into the
APK (`openzzs-core` static library); no prebuilt native library is required.

## Requirements

- JDK 17 or newer, installed as a **full JDK** (needs `jlink`; the stripped
  runtime bundled with some IDEs does not work)
- Android Studio with AGP 9.4 support, **or** a local Gradle 9.6+ install
- Android SDK Platform 37 and Build-Tools 36.0.0
- Android NDK 27.2.12479018 (pinned through `ndkVersion`; install it in the SDK manager)
- CMake 3.22.1 (installed through the SDK manager)
- A physical Android device running API 26+ (`minSdk = 26`)

The build uses AGP 9.4.1 with its built-in Kotlin support, so the standalone
`kotlin-android` plugin is intentionally absent. Gradle 9.6 is declared in the
committed wrapper (`gradle/wrapper/gradle-wrapper.properties`); use
`./gradlew` so the IDE and the command line share the same version.

> AGP 8.5 / Gradle 8.x is **not** compatible with this project anymore. If an
> older Android Studio is used, point its Gradle JDK at a full JDK 17+ and let
> the wrapper download Gradle 9.6.

## Configure

Create `local.properties` in this directory:

```properties
sdk.dir=/home/<user>/Android/Sdk
# Optional when several NDKs are installed:
# ndk.dir=/home/<user>/Android/Sdk/ndk/27.2.12479018
```

No API keys or signing configuration are needed for debug builds.

## Build and install

```bash
cd android

# Always use the wrapper so AGP and Gradle stay in sync:
./gradlew :app:assembleDebug

adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.openzzs/.MainActivity
```

Watch the native side:

```bash
adb logcat -s Openzzs-Sensr
```

## Using the app

The activity opens already streaming:

- **Start** — restarts the native sampler thread.
- **Stop** — stops it and leaves the queue statistics visible.
- **Flush** — discards queued events.
- **Sensors** — re-prints the sensor descriptors reported by the NDK.

Enabled by default (when present): accelerometer, gyroscope, magnetometer,
rotation vector, gravity, linear acceleration, light, proximity and step
counter, all at **50 Hz**.

The status line shows the backend, sensor count and the native lifetime
counters (`queued`, `enqueued`, `dropped`, `dequeued`, `produced`). A growing
`dropped` counter means the UI thread is polling slower than the device
produces; the queue capacity is 256 events and overflow drops the newest
sample by design.

## Permissions and notes

- Motion sensors need **no** runtime permission.
- `BODY_SENSORS` is declared for heart-rate sensors, but the sample does not
  enable heart rate. If you add it, request the permission at runtime first.
- The emulator exposes synthetic sensors; values are useful for smoke tests but
  not for validating real timestamps/accuracy. Prefer a physical device.
- `ZZS_std_heap_init()` is called once from `SensrBridge.initRuntime()` with an
  8 MiB heap; the runtime is torn down when the activity finishes.
- The automatic backend (`BACKEND_DEFAULT`) selects the NDK backend and only
  falls back to the deterministic mock backend when the NDK backend cannot be
  created.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `NoSuchMethodError: Project.exec(...)` during sync | Gradle 9 is running AGP 8.x. Use the committed wrapper (Gradle 9.6 + AGP 9.4.1) and delete stale `android/.gradle`/`android/.idea` caches. |
| `jlink executable ... does not exist` | Gradle is running on a stripped IDE runtime. Set **Settings → Build Tools → Gradle → Gradle JDK** to a full JDK 17+. |
| `SDK location not found` | Create `local.properties` as shown above. |
| `No version of NDK matched the requested version` | Install an NDK or set `ndkVersion`/`ndk.dir`. |
| `CMake 3.22.1 not found` | Install CMake 3.22.1 from the SDK manager. |
| `UnsatisfiedLinkError: openzzs-jni` | Ensure the ABI is enabled (default: all four) and clean/reinstall the app. |
| Log shows `sensors (0)` | The device exposes none of the requested types; `adb shell dumpsys sensorservice` lists what is available. |
