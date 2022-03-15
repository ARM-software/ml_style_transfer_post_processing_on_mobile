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

#include "acl_network.h"
#include <vector>
#include <memory>

/*
 * This helper class loads a tflite model file and adds layers to ACLNetwork one by one.
 * The weights are also loaded.
 *
 * Note: We are using 'Runtime' part of Arm Compute Library. It only provides individual functions/layers, so ACLNetwork class serves as a container.
 * This is why ACL does not have such functionality as parsing tflite files.
 * If you are interested in a higher level framework that supports multiple neural network file formats, take a look at ArmNN: https://github.com/ARM-software/armnn
 * We used ACL instead of ArmNN because we need to import OpenCL memory to tensors.
 * At the point of doing this experiment there was no such functionality in ArmNN, but it's planned in the future versions.
 */
class TFLiteParser
{
public:
    static std::unique_ptr<ACLNetwork> parse_model(const std::vector<uint8_t>& data, const arm_compute::CLTensor& input_output_tensor);
};