/* Copyright (c) 2022, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <arm_compute/runtime/CL/CLTensor.h>

// Transpose values order from tflite format to Arm Compute Library format for Conv2D and DepthwiseConv2D layers.
std::vector<float> transpose_kernel_values(const std::vector<float>& values,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t input_channels,
                                           uint32_t output_features);

// Transpose values order from tflite format to Arm Compute Library format for Conv2DTranspose layers.
std::vector<float> transpose_deconv_kernel_values(const std::vector<float>& values,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  uint32_t input_channels,
                                                  uint32_t output_features);

inline size_t get_tensor_offset(const arm_compute::ITensorInfo& info,
                                uint32_t depth_index,
                                uint32_t batch_index,
                                uint32_t channel_index,
                                uint32_t y,
                                uint32_t x);

inline size_t get_linear_buffer_offset(const arm_compute::ITensorInfo& info,
                                       uint32_t depth_index,
                                       uint32_t batch_index,
                                       uint32_t channel_index,
                                       uint32_t y,
                                       uint32_t x);

void copy_data_to_tensor(arm_compute::ITensor& tensor, const float* data);

void copy_data_from_tensor(const arm_compute::ITensor& tensor, float* data);

void set_tensor_values(arm_compute::CLTensor& tensor, const std::vector<float>& values);

void fill_tensor(arm_compute::CLTensor& tensor, float value);

std::vector<float> get_tensor_values(arm_compute::CLTensor& tensor);