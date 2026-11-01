# ProGuard rules for OpenZZS Android app
# https://developer.android.com/studio/build/shrink-code

# Keep the application class
-keep class com.openzzs.SensrBridge {
    public <methods>;
    public native <methods>;
}

# Keep JNI methods
-keepclassmembers class com.openzzs.SensrBridge {
    native <methods>;
}

# Keep classes that may be referenced by native code
-keep class com.openzzs.SensrActivity {
    <methods>;
}

# Keep native method names
-keepclasseswithmembers class * {
    native <methods>;
}

# Suppress warnings for OpenZZS native code
-dontwarn com.openzzs.**
