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
#include <arm_compute/runtime/CL/CLFunctions.h>

class ACLNetwork
{
public:
    ACLNetwork() = default;

    ~ACLNetwork() = default;

    ACLNetwork(const ACLNetwork&) = delete;

    ACLNetwork(ACLNetwork&&) = delete;

    void run();

    arm_compute::CLTensor& add_pad(const arm_compute::CLTensor& input, uint32_t pad_x, uint32_t pad_y);

    arm_compute::CLTensor& add_addition(const arm_compute::CLTensor& input_a,
                                        const arm_compute::CLTensor& input_b,
                                        arm_compute::ActivationLayerInfo::ActivationFunction activation);

    arm_compute::CLTensor& add_activation(const arm_compute::CLTensor& input,
                                          arm_compute::ActivationLayerInfo::ActivationFunction activation,
                                          float a = 0.0f,
                                          float b = 0.0f);

    arm_compute::CLTensor& add_conv2d(const arm_compute::CLTensor& input,
                                      uint32_t kernel_width,
                                      uint32_t kernel_height,
                                      uint32_t output_features,
                                      uint32_t pad_x_front,
                                      uint32_t pad_x_back,
                                      uint32_t pad_y_front,
                                      uint32_t pad_y_back,
                                      uint32_t stride_x,
                                      uint32_t stride_y,
                                      const std::vector<float>& kernel_values,
                                      const std::vector<float>& bias_values,
                                      arm_compute::ActivationLayerInfo::ActivationFunction activation,
                                      uint32_t dilation_x = 1,
                                      uint32_t dilation_y = 1);

    arm_compute::CLTensor& add_depthwise_conv2d(const arm_compute::CLTensor& input,
                                                uint32_t kernel_width,
                                                uint32_t kernel_height,
                                                uint32_t pad_x_front,
                                                uint32_t pad_x_back,
                                                uint32_t pad_y_front,
                                                uint32_t pad_y_back,
                                                uint32_t stride_x,
                                                uint32_t stride_y,
                                                const std::vector<float>& kernel_values,
                                                const std::vector<float>& bias_values,
                                                arm_compute::ActivationLayerInfo::ActivationFunction activation,
                                                uint32_t dilation_x = 1,
                                                uint32_t dilation_y = 1);

    arm_compute::CLTensor& add_conv2d_transpose(const arm_compute::CLTensor& input,
                                                uint32_t kernel_width,
                                                uint32_t kernel_height,
                                                uint32_t output_features,
                                                uint32_t pad_x_front,
                                                uint32_t pad_x_back,
                                                uint32_t pad_y_front,
                                                uint32_t pad_y_back,
                                                uint32_t stride_x,
                                                uint32_t stride_y,
                                                const std::vector<float>& kernel_values,
                                                const std::vector<float>& bias_values);

    // Additional layers that convert the image to sRGB color space.
    arm_compute::CLTensor& add_linear_to_srgb(const arm_compute::CLTensor& input);

    // Additinal layers that convert the image back to linear color space.
    arm_compute::CLTensor& add_srgb_to_linear(const arm_compute::CLTensor& input);

    arm_compute::CLTensor& add_dequantization(const arm_compute::CLTensor &input);

    void add_quantization(const arm_compute::CLTensor &input, const arm_compute::CLTensor &output);

    arm_compute::CLTensor& create_tensor(const std::vector<uint32_t>& dims);

private:
    std::vector<std::unique_ptr<arm_compute::CLTensor>> tensors;

    std::vector<std::unique_ptr<arm_compute::IFunction>> functions;
};