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

#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <core/image.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>
#include <CL/cl2.hpp>
#include "acl_utils/acl_network.h"

/*
 * Post-processing pipeline that uses Arm Compute Library (ACL) for running neural network inference.
 */
class ACLPipeline
{
public:
    ACLPipeline(uint32_t width, uint32_t height, uint32_t channels);

    void run(AHardwareBuffer* image_buffer, const VkExtent3D& extent);

private:
    cl::Context context;

    cl::CommandQueue queue;

    std::unique_ptr<ACLNetwork> net;

    std::unique_ptr<arm_compute::CLTensor> input_tensor;
};