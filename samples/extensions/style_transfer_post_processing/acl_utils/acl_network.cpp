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

#include "acl_network.h"
#include "tensor_utils.h"
#include "common/logging.h"

uint32_t calculate_conv_output_size(uint32_t input_size, uint32_t kernel_size, uint32_t pad, uint32_t stride, uint32_t dilation)
{
    return std::ceil((float)(input_size + 2 * pad - dilation * (kernel_size - 1)) / (float)(stride));
}

uint32_t calculate_deconv_output_size(uint32_t input_size, uint32_t kernel_size, uint32_t pad, uint32_t stride)
{
    return (input_size - 1) * stride - 2 * pad + kernel_size;
}

void ACLNetwork::run()
{
    for(const auto& function : functions)
    {
        function->prepare();
        function->run();
    }
}

arm_compute::CLTensor& ACLNetwork::create_tensor(const std::vector<uint32_t> &dims)
{
    arm_compute::TensorShape shape;
    for(int i = 0; i < dims.size(); i++)
    {
        shape.set(i, dims[i], false);
    }

    auto tensor = std::make_unique<arm_compute::CLTensor>();
    tensor->allocator()->init(arm_compute::TensorInfo(shape, 1, arm_compute::DataType::F32, arm_compute::DataLayout::NHWC));
    tensors.push_back(std::move(tensor));
    return *tensors.back();
}

arm_compute::CLTensor &ACLNetwork::add_addition(const arm_compute::CLTensor &input_a,
                                                const arm_compute::CLTensor &input_b,
                                                arm_compute::ActivationLayerInfo::ActivationFunction activation)
{
    auto input_shape = input_a.info()->tensor_shape();

    auto& output = create_tensor({(uint32_t)input_shape[0], (uint32_t)input_shape[1], (uint32_t)input_shape[2]});

    arm_compute::ActivationLayerInfo activation_info(activation);
    auto add = std::make_unique<arm_compute::CLArithmeticAddition>();
    add->configure((arm_compute::ICLTensor *) &input_a, (arm_compute::ICLTensor *) &input_b, &output, arm_compute::ConvertPolicy(), activation_info);
    functions.push_back(std::move(add));

    output.allocator()->allocate();

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_activation(const arm_compute::CLTensor &input,
                                                  arm_compute::ActivationLayerInfo::ActivationFunction activation,
                                                  float a,
                                                  float b)
{
    auto input_shape = input.info()->tensor_shape();

    auto& output = create_tensor({(uint32_t)input_shape[0], (uint32_t)input_shape[1], (uint32_t)input_shape[2]});

    auto add = std::make_unique<arm_compute::CLActivationLayer>();
    add->configure((arm_compute::ICLTensor *) &input, &output, arm_compute::ActivationLayerInfo(activation, a, b));
    functions.push_back(std::move(add));

    output.allocator()->allocate();

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_pad(const arm_compute::CLTensor &input, uint32_t pad_x, uint32_t pad_y)
{
    auto input_shape = input.info()->tensor_shape();
    uint32_t output_width = input_shape[1] + pad_x * 2;
    uint32_t output_height = input_shape[2] + pad_y * 2;

    arm_compute::PaddingList padding_list;
    padding_list.push_back(arm_compute::PaddingInfo{0, 0});
    padding_list.push_back(arm_compute::PaddingInfo{pad_x, pad_x});
    padding_list.push_back(arm_compute::PaddingInfo{pad_y, pad_y});

    auto& output = create_tensor({(uint32_t)input_shape[0], output_width, output_height});
    auto pad = std::make_unique<arm_compute::CLPadLayer>();
    pad->configure((arm_compute::ICLTensor *) &input, &output, padding_list);
    functions.push_back(std::move(pad));

    output.allocator()->allocate();
    return output;
}

arm_compute::CLTensor &ACLNetwork::add_conv2d(const arm_compute::CLTensor& input,
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
                                              uint32_t dilation_x,
                                              uint32_t dilation_y)
{
    arm_compute::TensorShape input_shape = input.info()->tensor_shape();

    uint32_t input_features = input_shape[0];
    uint32_t output_width = calculate_conv_output_size(input_shape[1], kernel_width,
                                                       std::max(pad_x_front, pad_x_back), stride_x,
                                                       dilation_x);
    uint32_t output_height = calculate_conv_output_size(input_shape[2], kernel_height,
                                                        std::max(pad_y_front, pad_y_back), stride_y,
                                                        dilation_y);

    auto& kernel = create_tensor({ input_features, kernel_width, kernel_height, output_features });
    auto& bias = create_tensor({output_features});
    auto& output = create_tensor({output_features, output_width, output_height});

    auto conv = std::make_unique<arm_compute::CLConvolutionLayer>();
    arm_compute::PadStrideInfo pad_stride_info(stride_x, stride_y, pad_x_front, pad_x_back, pad_y_front, pad_y_back, arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::WeightsInfo weights_info;
    arm_compute::Size2D dilation(dilation_x, dilation_y);
    arm_compute::ActivationLayerInfo activation_info(activation);

    auto status = conv->validate(input.info(), kernel.info(), bias.info(), output.info(), pad_stride_info, weights_info, dilation, activation_info);
    if(!status)
    {
        LOGE("Conv2D error, description: {}", status.error_description().c_str());
    }
    conv->configure((arm_compute::ICLTensor*)&input, &kernel, &bias, &output, pad_stride_info, weights_info, dilation, activation_info);
    functions.push_back(std::move(conv));

    output.allocator()->allocate();
    kernel.allocator()->allocate();
    bias.allocator()->allocate();

    set_tensor_values(kernel, kernel_values);
    set_tensor_values(bias, bias_values);

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_depthwise_conv2d(const arm_compute::CLTensor &input,
                                                        uint32_t kernel_width,
                                                        uint32_t kernel_height,
                                                        uint32_t pad_x_front,
                                                        uint32_t pad_x_back,
                                                        uint32_t pad_y_front,
                                                        uint32_t pad_y_back,
                                                        uint32_t stride_x,
                                                        uint32_t stride_y,
                                                        const std::vector<float> &kernel_values,
                                                        const std::vector<float> &bias_values,
                                                        arm_compute::ActivationLayerInfo::ActivationFunction activation,
                                                        uint32_t dilation_x,
                                                        uint32_t dilation_y)
{
    arm_compute::TensorShape input_shape = input.info()->tensor_shape();

    uint32_t input_features = input_shape[0];
    uint32_t output_width = calculate_conv_output_size(input_shape[1], kernel_width,
                                                       std::max(pad_x_front, pad_x_back), stride_x,
                                                       dilation_x);
    uint32_t output_height = calculate_conv_output_size(input_shape[2], kernel_height,
                                                        std::max(pad_y_front, pad_y_back), stride_y,
                                                        dilation_y);

    auto& kernel = create_tensor({ input_features, kernel_width, kernel_height });
    auto& bias = create_tensor({input_features});
    auto& output = create_tensor({input_features, output_width, output_height});

    auto conv = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer>();
    arm_compute::PadStrideInfo pad_stride_info(stride_x, stride_y, pad_x_front, pad_x_back, pad_y_front, pad_y_back, arm_compute::DimensionRoundingType::FLOOR);
    arm_compute::ActivationLayerInfo activation_info(activation);
    arm_compute::Size2D dilations(dilation_x, dilation_y);

    auto status = conv->validate(input.info(), kernel.info(), bias.info(), output.info(), pad_stride_info, 1, activation_info, dilations);
    if(!status)
    {
        LOGE("DepthwiseConv2D error, description: {}", status.error_description().c_str());
    }
    conv->configure((arm_compute::ICLTensor*)&input, &kernel, &bias, &output, pad_stride_info, 1, activation_info, dilations);
    functions.push_back(std::move(conv));

    output.allocator()->allocate();
    kernel.allocator()->allocate();
    bias.allocator()->allocate();

    set_tensor_values(kernel, kernel_values);
    set_tensor_values(bias, bias_values);

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_conv2d_transpose(const arm_compute::CLTensor &input,
                                                        uint32_t kernel_width,
                                                        uint32_t kernel_height,
                                                        uint32_t output_features,
                                                        uint32_t pad_x_front,
                                                        uint32_t pad_x_back,
                                                        uint32_t pad_y_front,
                                                        uint32_t pad_y_back,
                                                        uint32_t stride_x,
                                                        uint32_t stride_y,
                                                        const std::vector<float> &kernel_values,
                                                        const std::vector<float> &bias_values)
{
    arm_compute::TensorShape input_shape = input.info()->tensor_shape();

    uint32_t input_features = input_shape[0];
    uint32_t output_width = calculate_deconv_output_size(input_shape[1], kernel_width,
                                                         std::max(pad_x_front, pad_x_back),
                                                         stride_x);
    uint32_t output_height = calculate_deconv_output_size(input_shape[2], kernel_height,
                                                          std::max(pad_y_front, pad_y_back),
                                                          stride_y);

    auto& kernel = create_tensor({ input_features, kernel_width, kernel_height, output_features });
    auto& bias = create_tensor({output_features});
    auto& output = create_tensor({output_features, output_width, output_height});

    auto deconv = std::make_unique<arm_compute::CLDeconvolutionLayer>();
    arm_compute::PadStrideInfo pad_stride_info(stride_x, stride_y, pad_x_front, pad_x_back, pad_y_front, pad_y_back, arm_compute::DimensionRoundingType::FLOOR);

    auto status = deconv->validate(input.info(), kernel.info(), bias.info(), output.info(), pad_stride_info);
    if(!status)
    {
        LOGE("Conv2DTranspose error, description: {}", status.error_description().c_str());
    }
    deconv->configure((arm_compute::ICLTensor *) &input, &kernel, &bias, &output, pad_stride_info);
    functions.push_back(std::move(deconv));

    output.allocator()->allocate();
    kernel.allocator()->allocate();
    bias.allocator()->allocate();

    set_tensor_values(kernel, kernel_values);
    set_tensor_values(bias, bias_values);

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_dequantization(const arm_compute::CLTensor &input)
{
    auto input_shape = input.info()->tensor_shape();

    auto& output = create_tensor({(uint32_t)input_shape[0], (uint32_t)input_shape[1], (uint32_t)input_shape[2]});
    auto dequantization = std::make_unique<arm_compute::CLDequantizationLayer>();
    dequantization->configure(&input, &output);
    functions.push_back(std::move(dequantization));

    output.allocator()->allocate();
    return output;
}

void ACLNetwork::add_quantization(const arm_compute::CLTensor &input, const arm_compute::CLTensor &output)
{
    auto quantization = std::make_unique<arm_compute::CLQuantizationLayer>();
    quantization->configure(&input, (arm_compute::ICLTensor *) &output);
    functions.push_back(std::move(quantization));
}

arm_compute::CLTensor &ACLNetwork::add_linear_to_srgb(const arm_compute::CLTensor &input)
{
    // We first need to normalize values to [0, 1] (during this step we also multiply all the values by 'brightness_adjustment' to make the image brighter).
    // Then the values are calculated as (x ** (1 / 2.4)) * 269.025 - 14.025. These values are again in range [0, 255].

    arm_compute::TensorShape input_shape = input.info()->tensor_shape();

    float brightness_adjustment = 1.7f;
    auto& input_normalized = add_activation(input, arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR, 1.0f / 255.0f * brightness_adjustment, 0.0f);

    auto& output = create_tensor({(uint32_t)input_shape[0], (uint32_t)input_shape[1], (uint32_t)input_shape[2]});
    auto& multiplier = create_tensor({1});

    arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR, 269.025, -14.025);

    auto elementwise_pow = std::make_unique<arm_compute::CLElementwisePower>();
    auto status = elementwise_pow->validate(input_normalized.info(), multiplier.info(), output.info(), act_info);
    if(!status)
    {
        LOGE("ElementwisePower error, description: {}", status.error_description().c_str());
    }
    elementwise_pow->configure((arm_compute::ICLTensor *) &input_normalized, &multiplier, &output, act_info);
    functions.push_back(std::move(elementwise_pow));

    multiplier.allocator()->allocate();
    output.allocator()->allocate();

    set_tensor_values(multiplier, {1.0f / 2.4f});

    return output;
}

arm_compute::CLTensor &ACLNetwork::add_srgb_to_linear(const arm_compute::CLTensor &input)
{
    // We first need to normalize values to [0, 1].
    // Then the values are calculated as (((x + 0.055) / 1.055) ** 2.4) * 255. These values are again in range [0, 255].

    arm_compute::TensorShape input_shape = input.info()->tensor_shape();
    auto& output = create_tensor({(uint32_t)input_shape[0], (uint32_t)input_shape[1], (uint32_t)input_shape[2]});

    auto& input_normalized = add_activation(input, arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR, 1.0f / 255.0f, 0.055f);
    auto& input_divided = add_activation(input, arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR, 1.0f / 1.055f, 0.0f);

    arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::LINEAR, 255.0f, 0);

    auto& multiplier = create_tensor({1});
    auto elementwise_pow = std::make_unique<arm_compute::CLElementwisePower>();
    auto status = elementwise_pow->validate(input_normalized.info(), multiplier.info(), output.info(), act_info);
    if(!status)
    {
        LOGE("ElementwisePower error, description: {}", status.error_description().c_str());
    }
    elementwise_pow->configure((arm_compute::ICLTensor *) &input_normalized, &multiplier, &output, act_info);
    functions.push_back(std::move(elementwise_pow));

    multiplier.allocator()->allocate();
    output.allocator()->allocate();

    set_tensor_values(multiplier, {2.4f});

    return output;
}
