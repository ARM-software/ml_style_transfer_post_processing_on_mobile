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

#include "acl_pipeline.h"
#include <platform/filesystem.h>
#include <platform/platform.h>
#include "acl_utils/tflite_parser.h"

ACLPipeline::ACLPipeline(uint32_t width, uint32_t height, uint32_t channels)
{
    arm_compute::CLScheduler::get().default_init();
    context = arm_compute::CLScheduler::get().context();
    queue = arm_compute::CLScheduler::get().queue();

    input_tensor = std::make_unique<arm_compute::CLTensor>();

    arm_compute::Strides strides(1, channels, width * channels);
    arm_compute::CLTensor input;
    arm_compute::TensorShape input_shape(3, width, height);
    arm_compute::QuantizationInfo quantization_info(1.0f / 1.0f, 0);
    arm_compute::TensorInfo tensor_info;
    size_t total_size = width * height * channels;
    tensor_info.init(input_shape, 1, arm_compute::DataType::QASYMM8, strides, 0, total_size);
    tensor_info.set_quantization_info(quantization_info);
    tensor_info.set_data_layout(arm_compute::DataLayout::NHWC);
    input_tensor->allocator()->init(tensor_info);

    auto model_data = vkb::fs::read_asset("nn_models/style_transfer.tflite");
    net = TFLiteParser::parse_model(model_data, *input_tensor);
}

cl_mem import_hardware_buffer_to_opencl(cl_context context, AHardwareBuffer* hardware_buffer)
{
    cl_int error = CL_SUCCESS;
    const cl_import_properties_arm cl_import_properties[] = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM, 0 };
    cl_mem imported_memory = clImportMemoryARM(context,
                                               CL_MEM_READ_WRITE,
                                               cl_import_properties,
                                               hardware_buffer,
                                               CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM,
                                               &error);

    if(error != CL_SUCCESS )
    {
        throw std::runtime_error("Cannot import hardware buffer. Error: " + std::to_string(error));
    }
    return imported_memory;
}

void ACLPipeline::run(AHardwareBuffer* image_buffer, const VkExtent3D& extent)
{
    // First we import AHardwareBuffer into OpenCL using clImportMemoryARM.
    auto imported_memory = import_hardware_buffer_to_opencl(context.get(), image_buffer);
    cl::Buffer input_buffer(imported_memory);

    // Then we can specify imported OpenCL memory as memory for an ACL tensor.
    auto status = input_tensor->allocator()->import_memory(input_buffer);
    if(!status)
    {
        throw std::runtime_error("Failed to import CLTensor memory, Error: " + status.error_description());
    }

    net->run();

    arm_compute::CLScheduler::get().queue().flush();
    arm_compute::CLScheduler::get().queue().finish();
}

