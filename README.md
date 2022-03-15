<!--
- Copyright (c) 2019-2022, Arm Limited and Contributors
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

This project demonstrates how to use [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) with Vulkan to achieve ML-based style transfer post processing of rendered frames. 
For more info take a look at our blog on this topic: [Style transfer as graphics post-processing on mobile](https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/style-transfer-on-mobile).

The source code is based on [Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) project maintained by Khronos Group Inc.

You can find more info on using Vulkan and OpenCL extensions to achieve Vulkan-OpenCL interoperability in the corresponding Vulkan [code sample](https://github.com/KhronosGroup/Vulkan-Samples/tree/master/samples/extensions/open_cl_interop).

The code of the demo application can be found in [samples/extensions/style_transfer_post_processing](./samples/extensions/style_transfer_post_processing) directory.

Additionaly you can find a jupyter notebook that demonstrates how to build and train the style transfer neural network used in the demo app (see the [network](./network) directory).

## Requirements

To run the demo you need a device with OpenCL support. Apart from that, the following OpenCL and Vulkan extensions must be supported:

- cl_arm_import_memory
- VK_KHR_external_memory
- VK_ANDROID_external_memory_android_hardware_buffer

## Setup

Clone the repo with submodules using the following command:

```
git clone --recurse-submodules https://github.com/ARM-software/ml_style_transfer_post_processing_on_mobile.git
cd ml_style_transfer_post_processing_on_mobile
```

Follow build instructions for below.

## Build

Android is the only supported platform. For instructions refer to the [build guide](./docs/build.md#android "Android Build Guide").

## License

See [LICENSE](LICENSE).

This project has some third-party dependencies, each of which may have independent licensing:

- [astc-encoder](https://github.com/ARM-software/astc-encoder): ASTC Evaluation Codec
- [CTPL](https://github.com/vit-vit/CTPL): Thread Pool Library
- [docopt](https://github.com/docopt/docopt.cpp): A C++11 port of the Python argument parsing library
- [glfw](https://github.com/glfw/glfw): A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input
- [glm](https://github.com/g-truc/glm): OpenGL Mathematics
- [glslang](https://github.com/KhronosGroup/glslang): Shader front end and validator
- [dear imgui](https://github.com/ocornut/imgui): Immediate Mode Graphical User Interface
- [HWCPipe](https://github.com/ARM-software/HWCPipe): Interface to mobile Hardware Counters
- [KTX-Software](https://github.com/KhronosGroup/KTX-Software): Khronos Texture Library and Tools
- [spdlog](https://github.com/gabime/spdlog): Fast C++ logging library
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross): Parses and converts SPIR-V to other shader languages
- [stb](https://github.com/nothings/stb): Single-file public domain (or MIT licensed) libraries
- [tinygltf](https://github.com/syoyo/tinygltf): Header only C++11 glTF 2.0 file parser
- [nlohmann json](https://github.com/nlohmann/json): C++ JSON Library (included by [tinygltf](https://github.com/syoyo/tinygltf))
- [vma](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator): Vulkan Memory Allocator
- [volk](https://github.com/zeux/volk): Meta loader for Vulkan API
- [vulkan](https://github.com/KhronosGroup/Vulkan-Docs): Sources for the formal documentation of the Vulkan API
- [flatbuffers](https://github.com/google/flatbuffers): Serialization library that is used for parsing a neural network model in tflite format
- [tensorflow](https://github.com/tensorflow/tensorflow): Machine learning framework, used to generate [tflite schema](./third_party/tflite_schema/tflite_schema.h) for parsing a neural network from a file

This project uses assets from [vulkan-samples-assets](https://github.com/KhronosGroup/Vulkan-Samples-Assets). Each one has its own license.

### Trademarks

Vulkan is a registered trademark of the Khronos Group Inc.
