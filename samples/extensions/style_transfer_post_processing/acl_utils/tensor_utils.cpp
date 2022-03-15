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

#include "tensor_utils.h"

std::vector<float> transpose_kernel_values(const std::vector<float>& values,
                                           uint32_t width,
                                           uint32_t height,
                                           uint32_t input_channels,
                                           uint32_t output_features)
{
    std::vector<float> transposed_values(values.size());
    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < input_channels; c++)
            {
                for (int f = 0; f < output_features; f++)
                {
                    int src_index = ((y * width + x) * input_channels + c) * output_features + f;
                    int dst_index = ((f * height + y) * width + x) * input_channels + c;
                    transposed_values[dst_index] = values[src_index];
                }
            }
        }
    }
    return transposed_values;
}

std::vector<float> transpose_deconv_kernel_values(const std::vector<float>& values,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  uint32_t input_channels,
                                                  uint32_t output_features)
{
    std::vector<float> transposed_values(values.size());
    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < input_channels; c++)
            {
                for (int f = 0; f < output_features; f++)
                {
                    int src_index = ((y * width + x) * input_channels + c) * output_features + f;
                    int dst_index = ((c * height + y) * width + x) * output_features + f;
                    transposed_values[dst_index] = values[src_index];
                }
            }
        }
    }
    return transposed_values;
}

inline size_t get_tensor_offset(const arm_compute::ITensorInfo& info,
                                uint32_t depth_index,
                                uint32_t batch_index,
                                uint32_t channel_index,
                                uint32_t y,
                                uint32_t x)
{
    arm_compute::Coordinates coords;
    coords.set(4, static_cast<int>(depth_index));
    coords.set(3, static_cast<int>(batch_index));
    coords.set(2, static_cast<int>(channel_index));
    coords.set(1, static_cast<int>(y));
    coords.set(0, static_cast<int>(x));
    return (size_t)(info.offset_element_in_bytes(coords));
}

inline size_t get_linear_buffer_offset(const arm_compute::ITensorInfo& info,
                                       uint32_t depth_index,
                                       uint32_t batch_index,
                                       uint32_t channel_index,
                                       uint32_t y,
                                       uint32_t x)
{
    const arm_compute::TensorShape& shape = info.tensor_shape();
    uint32_t width = static_cast<uint32_t>(shape[0]);
    uint32_t height = static_cast<uint32_t>(shape[1]);
    uint32_t num_channels = static_cast<uint32_t>(shape[2]);
    uint32_t num_batches = static_cast<uint32_t>(shape[3]);
    return (((depth_index * num_batches + batch_index) * num_channels + channel_index) * height + y) * width + x;
}

void copy_data_to_tensor(arm_compute::ITensor& tensor, const float* data)
{
    auto& info = *tensor.info();
    auto& shape = info.tensor_shape();

    uint8_t* buffer_ptr = tensor.buffer();
    uint32_t width = static_cast<uint32_t>(shape[0]);
    uint32_t height = static_cast<uint32_t>(shape[1]);
    uint32_t num_channels = static_cast<uint32_t>(shape[2]);
    uint32_t num_batches = static_cast<uint32_t>(shape[3]);
    uint32_t depth = static_cast<uint32_t>(shape[4]);

    for (unsigned int depth_index = 0; depth_index < depth; ++depth_index)
    {
        for (unsigned int batch_index = 0; batch_index < num_batches; ++batch_index)
        {
            for (unsigned int channel_index = 0; channel_index < num_channels; ++channel_index)
            {
                for (unsigned int y = 0; y < height; ++y)
                {
                    memcpy(buffer_ptr + get_tensor_offset(info, depth_index, batch_index, channel_index, y, 0),
                           data + get_linear_buffer_offset(info, depth_index, batch_index, channel_index, y, 0),
                           width * sizeof(float));
                }
            }
        }
    }
}

void copy_data_from_tensor(const arm_compute::ITensor& tensor, float* data)
{
    const auto& info = *tensor.info();
    const auto& shape = info.tensor_shape();

    const uint8_t* buffer_ptr = tensor.buffer();
    uint32_t width = static_cast<uint32_t>(shape[0]);
    uint32_t height = static_cast<uint32_t>(shape[1]);
    uint32_t num_channels = static_cast<uint32_t>(shape[2]);
    uint32_t num_batches = static_cast<uint32_t>(shape[3]);
    uint32_t depth = static_cast<uint32_t>(shape[4]);

    for (unsigned int depth_index = 0; depth_index < depth; ++depth_index)
    {
        for (unsigned int batch_index = 0; batch_index < num_batches; ++batch_index)
        {
            for (unsigned int channel_index = 0; channel_index < num_channels; ++channel_index)
            {
                for (unsigned int y = 0; y < height; ++y)
                {
                    memcpy(data + get_linear_buffer_offset(info, depth_index, batch_index, channel_index, y, 0),
                           buffer_ptr + get_tensor_offset(info, depth_index, batch_index, channel_index, y,0),
                           width * sizeof(float));
                }
            }
        }
    }
}

void set_tensor_values(arm_compute::CLTensor& tensor, const std::vector<float>& values)
{
    tensor.map();
    copy_data_to_tensor(tensor, values.data());
    tensor.unmap();
}

void fill_tensor(arm_compute::CLTensor& tensor, float value)
{
    std::vector<float> values(tensor.info()->tensor_shape().total_size(), value);
    set_tensor_values(tensor, values);
}

std::vector<float> get_tensor_values(arm_compute::CLTensor& tensor)
{
    size_t num_elements = tensor.info()->tensor_shape().total_size();
    std::vector<float> values(num_elements);
    tensor.map();
    copy_data_from_tensor(tensor, values.data());
    tensor.unmap();
    return values;
}