<!--
- Copyright (c) 2019-2021, Arm Limited and Contributors
-
- SPDX-License-Identifier: Apache-2.0
-
- Licensed under the Apache License, Version 2.0 the "License";
- you may not use this file except in compliance with the License.
- You may obtain a copy of the License at
-
-     http://www.apache.org/licenses/LICENSE-2.0
-
- Unless required by applicable law or agreed to in writing, software
- distributed under the License is distributed on an "AS IS" BASIS,
- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
- See the License for the specific language governing permissions and
- limitations under the License.
-
-->

# Build Guides <!-- omit in toc -->

# Contents <!-- omit in toc -->

- [Resources](#resources)
- [Performance data](#performance-data)
- [Android Build](#android-build)
  - [Dependencies](#dependencies-3)
  - [Build with Gradle](#build-with-gradle)
  - [Build with Android Studio](#build-with-android-studio)
  - [Known Issues](#known-issues)

# Resources

The app is reading assets and shaders from the external storage. 

On Android, Gradle will run CMake which will sync assets and shaders to the device if there has been a change.

However, to sync them manually you may run the following command to ensure up to date assets are on the device:

```
adb push --sync assets /sdcard/Android/data/com.arm.style_transfer_post_processing/files/
adb push --sync shaders /sdcard/Android/data/com.arm.style_transfer_post_processing/files/
```

# Performance data

In order for performance data to be displayed, profiling needs to be enabled on the device. Some devices may disable it by default.

Profiling can be enabled via adb:

```
adb shell setprop security.perf_harden 0
```

> Performance data is captured using HWCPipe.
> For details on this project and how to integrate it in your pipeline,
> visit: https://github.com/ARM-software/HWCPipe

# Android Build

## Dependencies

For all dependencies set the following environment variables.

- CMake v3.10+
- JDK 8+ `JAVA_HOME=<SYSTEM_DIR>/java`
- Android NDK r18+ `ANDROID_NDK_ROOT=<WORK_DIR>/android-ndk`
- Android SDK `ANDROID_HOME=<WORK_DIR>/android-sdk`
- Gradle 5+ `GRADLE_HOME=<WORK_DIR>/gradle`
- [CMake Options](#cmake-options)
- [3D models](#3d-models)
- [Performance data](#performance-data)

The project was tested with the following environment:
- Android Studio 2020.3.1 Patch 3
- Android SDK 31
- Android NDK 23.0.7599858
- CMake v3.18
- Gradle 7.0.2

## Build with Gradle

### Generate the gradle project 

Use the provided script for the platform you are building on by running the following command:

#### Windows <!-- omit in toc -->

```
bldsys\scripts\generate_android_gradle.bat
```

#### Linux <!-- omit in toc -->

```
./bldsys/scripts/generate_android_gradle.sh
```

A new folder will be created in the root directory at `build\android_gradle`

### Build the project


```
cd build/android_gradle
```

> Prefer a release build for better performance unless you want to actively debug the application.

For a release build:

```
gradle assembleRelease
``` 

For a debug build:

```
gradle assembleDebug
``` 

### Install the apk on the device

You can now install the apk on a connected device using the Android Debug Bridge:

For a release build:

```
adb install build/outputs/apk/release/vulkan_samples-release.apk
```
For a debug build:

```
adb install build/outputs/apk/debug/vulkan_samples-debug.apk
```

## Build with Android Studio

> Alternatively, you may import the `build/android_gradle` folder in Android Studio and run the project from here

If you are using a newer version of cmake then 3.13, you might get this error:

> Execution failed for task ':externalNativeBuildDebug'.
Expected output file at \<PATH> for target \<sample> but there was none

In this case, update the version of the gradle plugin in "bldsys/cmake/template/gradle/build.gradle.in" to 3.5.0, remove the content of build folder and repeat the build process from Step 1. This is known to work with Gradle 6.3 and NDK 20.0.55

If you are using Android Studio, you can simply do these changes after importing the `build/android_gradle` folder, opening File->Project Structure, and doing the following changes:
On the Project tab, change the Android Gradle Plugin version to 3.5.0 and the Gradle version to 6.3.(this also requires NDK 20.0.55)

## Known issues

Depending on the environment (different Android Studio, Gradle, CMake versions) there can be difficulties building the project. In case of build errors please refer to Vulkan Samples [issue tracker](https://github.com/KhronosGroup/Vulkan-Samples/issues).

### Issue #1

```
AAPT: error: attribute android:requestLegacyExternalStorage not found.
```
Please make sure that `compileSdk` and `targetSdk` in build.gradle are specified as 29 or higher.

### Issue #2

```
NDK is not installed
```
Use SDK manager to install NDK r18 or higher. Then make sure the `ndkVersion` specified in build.gradle matches the installed version (alternatively assign full path to NDK to `ndk.dir` in local.properties).  

## Issue #3

```
Execution failed for task ':compileDebugJavaWithJavac'.
```

Make sure that you are using Java 11, `compileSdk` and `targetSdk` in build.gradle are specified as 29 or higher, and gradle as well as gradle-plugin versions are set to 7.0.2.

## Issue #4

```
Error Message: clGetPlatformIDs
```
Any OpenCL related error messages mean that either OpenCL or the required extension (`cl_arm_import_memory`) is not supported.