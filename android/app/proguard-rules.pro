# Keep the JNI facade and every native method reachable from Java.
-keep class com.openzzs.SensrBridge { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
