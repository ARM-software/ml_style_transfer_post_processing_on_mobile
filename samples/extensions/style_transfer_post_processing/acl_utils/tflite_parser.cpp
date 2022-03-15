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

#include "tflite_parser.h"

#include <tflite_schema.h>
#include <common/logging.h>
#include <platform/filesystem.h>
#include <platform/platform.h>
#include "tensor_utils.h"

using ActivationFunction = arm_compute::ActivationLayerInfo::ActivationFunction;

const static std::unordered_map<int32_t, ActivationFunction> TFLITE_TO_ACL_ACTIVATION =
{
    {tflite::ActivationFunctionType_NONE, ActivationFunction::IDENTITY },
    {tflite::ActivationFunctionType_RELU, ActivationFunction::RELU}
};

std::vector<uint32_t> to_uint_vector(const flatbuffers::Vector<int32_t>* int_vector)
{
    std::vector<uint32_t> uint_vector;
    for(auto i : *int_vector)
    {
        uint_vector.push_back((uint32_t)i);
    }
    return uint_vector;
}

void calculate_padding(uint32_t input_size,
                 uint32_t kernel_size,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t& padding_front,
                 uint32_t& padding_back,
                 tflite::Padding padding)
{
    padding_front = 0;
    padding_back = 0;
    if (padding == tflite::Padding_SAME)
    {
        uint32_t output_size = (input_size + stride - 1) / stride;
        uint32_t dilated_size = kernel_size + (dilation - 1) * (kernel_size - 1);
        uint32_t temp = (output_size - 1) * stride + dilated_size;
        if (temp > input_size)
        {
            padding_front = (temp - input_size) / 2;
            padding_back = (temp - input_size) - padding_front;
        }
    }
}

std::vector<float> copy_to_vector(const float* values, size_t size)
{
    std::vector<float> values_vector(size / sizeof(float));
    memcpy(values_vector.data(), values, size);
    return values_vector;
}

void parse_transpose_conv_2d(ACLNetwork& net,
                   std::unordered_map<int32_t, arm_compute::CLTensor*>& tensors,
                   const tflite::Model& model,
                   const tflite::SubGraph& subgraph,
                   const tflite::Operator& op)
{
    auto options = op.builtin_options_as_TransposeConvOptions();
    auto& input_indices = *op.inputs();
    auto& output_indices = *op.outputs();

    const auto& input = tensors.at(input_indices.Get(2));
    auto input_shape = input->info()->tensor_shape();
    uint32_t input_width = input_shape[1];
    uint32_t input_height = input_shape[2];

    const auto& kernel_tensor = subgraph.tensors()->Get(input_indices.Get(1));
    const auto& kernel_buffer = model.buffers()->Get(kernel_tensor->buffer());
    auto kernel_shape = to_uint_vector(kernel_tensor->shape());

    const auto& bias_tensor = subgraph.tensors()->Get(input_indices.Get(3));
    const auto& bias_buffer = model.buffers()->Get(bias_tensor->buffer());

    uint32_t kernel_width = kernel_shape[2];
    uint32_t kernel_height = kernel_shape[1];
    uint32_t output_features = kernel_shape[0];
    int32_t stride_x = (int32_t)options->stride_w();
    int32_t stride_y = (int32_t)options->stride_h();
    uint32_t padding_front_x = 0;
    uint32_t padding_back_x = 0;
    uint32_t padding_front_y = 0;
    uint32_t padding_back_y = 0;

    calculate_padding(input_width, kernel_width, stride_x, 1, padding_front_x, padding_back_x, options->padding());
    calculate_padding(input_height, kernel_height, stride_y, 1, padding_front_y, padding_back_y, options->padding());

    auto kernel_values = copy_to_vector((const float*)kernel_buffer->data()->Data(), kernel_buffer->data()->size());
    auto bias_values = copy_to_vector((const float*)bias_buffer->data()->Data(), bias_buffer->data()->size());

    tensors[output_indices.Get(0)] = &net.add_conv2d_transpose(*input,
                                                     kernel_width,
                                                     kernel_height,
                                                     output_features,
                                                     padding_front_x,
                                                     padding_back_x,
                                                     padding_front_y,
                                                     padding_back_y,
                                                     stride_x,
                                                     stride_y,
                                                     kernel_values,
                                                     bias_values);
}

void parse_depthwise_conv_2d(ACLNetwork& net,
                   std::unordered_map<int32_t, arm_compute::CLTensor*>& tensors,
                   const tflite::Model& model,
                   const tflite::SubGraph& subgraph,
                   const tflite::Operator& op)
{
    auto options = op.builtin_options_as_DepthwiseConv2DOptions();
    auto& input_indices = *op.inputs();
    auto& output_indices = *op.outputs();

    const auto& input = tensors.at(input_indices.Get(0));
    auto input_shape = input->info()->tensor_shape();
    uint32_t input_width = input_shape[1];
    uint32_t input_height = input_shape[2];

    const auto& kernel_tensor = subgraph.tensors()->Get(input_indices.Get(1));
    const auto& kernel_buffer = model.buffers()->Get(kernel_tensor->buffer());
    auto kernel_shape = to_uint_vector(kernel_tensor->shape());

    const auto& bias_tensor = subgraph.tensors()->Get(input_indices.Get(2));
    const auto& bias_buffer = model.buffers()->Get(bias_tensor->buffer());

    uint32_t kernel_width = kernel_shape[2];
    uint32_t kernel_height = kernel_shape[1];
    int32_t stride_x = (int32_t)options->stride_w();
    int32_t stride_y = (int32_t)options->stride_h();
    int32_t dilation_x = options->dilation_w_factor();
    int32_t dilation_y = options->dilation_h_factor();
    uint32_t padding_front_x = 0;
    uint32_t padding_back_x = 0;
    uint32_t padding_front_y = 0;
    uint32_t padding_back_y = 0;

    calculate_padding(input_width, kernel_width, stride_x, dilation_x, padding_front_x, padding_back_x, options->padding());
    calculate_padding(input_height, kernel_height, stride_y, dilation_y, padding_front_y, padding_back_y, options->padding());

    auto activation_function = TFLITE_TO_ACL_ACTIVATION.at(options->fused_activation_function());

    auto kernel_values = copy_to_vector((const float*)kernel_buffer->data()->Data(), kernel_buffer->data()->size());
    auto bias_values = copy_to_vector((const float*)bias_buffer->data()->Data(), bias_buffer->data()->size());

    tensors[output_indices.Get(0)] = &net.add_depthwise_conv2d(*input,
                                                     kernel_width,
                                                     kernel_height,
                                                     padding_front_x,
                                                     padding_back_x,
                                                     padding_front_y,
                                                     padding_back_y,
                                                     stride_x,
                                                     stride_y,
                                                     kernel_values,
                                                     bias_values,
                                                     activation_function,
                                                     dilation_x,
                                                     dilation_y);
}

void parse_conv_2d(ACLNetwork& net,
                   std::unordered_map<int32_t, arm_compute::CLTensor*>& tensors,
                   const tflite::Model& model,
                   const tflite::SubGraph& subgraph,
                   const tflite::Operator& op)
{
    auto options = op.builtin_options_as_Conv2DOptions();
    auto& input_indices = *op.inputs();
    auto& output_indices = *op.outputs();

    const auto& input = tensors.at(input_indices.Get(0));
    auto input_shape = input->info()->tensor_shape();
    uint32_t input_width = input_shape[1];
    uint32_t input_height = input_shape[2];

    const auto& kernel_tensor = subgraph.tensors()->Get(input_indices.Get(1));
    const auto& kernel_buffer = model.buffers()->Get(kernel_tensor->buffer());
    auto kernel_shape = to_uint_vector(kernel_tensor->shape());

    const auto& bias_tensor = subgraph.tensors()->Get(input_indices.Get(2));
    const auto& bias_buffer = model.buffers()->Get(bias_tensor->buffer());

    uint32_t kernel_width = kernel_shape[2];
    uint32_t kernel_height = kernel_shape[1];
    uint32_t output_features = kernel_shape[0];
    int32_t stride_x = (int32_t)options->stride_w();
    int32_t stride_y = (int32_t)options->stride_h();
    int32_t dilation_x = options->dilation_w_factor();
    int32_t dilation_y = options->dilation_h_factor();
    uint32_t padding_front_x = 0;
    uint32_t padding_back_x = 0;
    uint32_t padding_front_y = 0;
    uint32_t padding_back_y = 0;

    calculate_padding(input_width, kernel_width, stride_x, dilation_x, padding_front_x, padding_back_x, options->padding());
    calculate_padding(input_height, kernel_height, stride_y, dilation_y, padding_front_y, padding_back_y, options->padding());

    auto activation_function = TFLITE_TO_ACL_ACTIVATION.at(options->fused_activation_function());

    auto kernel_values = copy_to_vector((const float*)kernel_buffer->data()->Data(), kernel_buffer->data()->size());
    auto bias_values = copy_to_vector((const float*)bias_buffer->data()->Data(), bias_buffer->data()->size());

    tensors[output_indices.Get(0)] = &net.add_conv2d(*input,
                   kernel_width,
                   kernel_height,
                   output_features,
                   padding_front_x,
                   padding_back_x,
                   padding_front_y,
                   padding_back_y,
                   stride_x,
                   stride_y,
                   kernel_values,
                   bias_values,
                   activation_function,
                   dilation_x,
                   dilation_y);
}

void parse_relu(ACLNetwork& net,
                std::unordered_map<int32_t, arm_compute::CLTensor*>& tensors,
                const tflite::Model& model,
                const tflite::SubGraph& subgraph,
                const tflite::Operator& op)
{
    auto& input_indices = *op.inputs();
    auto& output_indices = *op.outputs();

    const auto& input = tensors.at(input_indices.Get(0));

    tensors[output_indices.Get(0)] = &net.add_activation(*input, ActivationFunction::RELU);
}

void parse_add(ACLNetwork& net,
                std::unordered_map<int32_t, arm_compute::CLTensor*>& tensors,
                const tflite::Model& model,
                const tflite::SubGraph& subgraph,
                const tflite::Operator& op)
{
    auto options = op.builtin_options_as_AddOptions();
    auto& input_indices = *op.inputs();
    auto& output_indices = *op.outputs();

    const auto& input0 = tensors.at(input_indices.Get(0));
    const auto& input1 = tensors.at(input_indices.Get(1));

    auto activation_function = TFLITE_TO_ACL_ACTIVATION.at(options->fused_activation_function());

    tensors[output_indices.Get(0)] = &net.add_addition(*input0, *input1, activation_function);
}

std::unique_ptr<ACLNetwork> TFLiteParser::parse_model(const std::vector<uint8_t> &data,  const arm_compute::CLTensor &input_output_tensor)
{
    auto network = std::make_unique<ACLNetwork>();
    auto &input_model = *tflite::GetModel(data.data());
    auto &input_subgraphs = *input_model.subgraphs();
    auto &subgraph = *input_subgraphs.Get(0);

    const auto& opcodes = *input_model.operator_codes();
    std::unordered_map<int32_t, arm_compute::CLTensor*> tensors;

    auto input_indices = to_uint_vector(subgraph.inputs());
    auto output_indices = to_uint_vector(subgraph.outputs());
    if(input_indices.size() > 1 || output_indices.size() > 1)
    {
        throw std::runtime_error("The model has more than one input/output, but single input/output tensor is specified.");
    }

    auto& dequantized_input = network->add_dequantization(input_output_tensor);

    // The model is intended to be used with rendered images in linear color space.
    // We are adding conversion to sRGB to improve quality when the images are processed using a neural network.
    tensors[input_indices[0]] = &network->add_linear_to_srgb(dequantized_input);

    for(const auto& op : *subgraph.operators())
    {
        uint32_t opcode_index = op->opcode_index();
        const auto& opcode = *opcodes[opcode_index];
        auto builtin_code = opcode.deprecated_builtin_code();

        switch (builtin_code)
        {
            case tflite::BuiltinOperator_CONV_2D:
                parse_conv_2d(*network, tensors, input_model, subgraph, *op);
                break;
            case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
                parse_depthwise_conv_2d(*network, tensors, input_model, subgraph, *op);
                break;
            case tflite::BuiltinOperator_RELU:
                parse_relu(*network, tensors, input_model, subgraph, *op);
                break;
            case tflite::BuiltinOperator_ADD:
                parse_add(*network, tensors, input_model, subgraph, *op);
                break;
            case tflite::BuiltinOperator_TRANSPOSE_CONV:
                parse_transpose_conv_2d(*network, tensors, input_model, subgraph, *op);
                break;
            default:
                throw std::runtime_error("Operation with builtin code " + std::to_string(builtin_code) + " is not supported by tflite importer.");
        }
    }

    // Converting the result back to linear color space.
    auto& linear_output = network->add_srgb_to_linear(*tensors[output_indices[0]]);
    network->add_quantization(linear_output, input_output_tensor);

    return network;
}
