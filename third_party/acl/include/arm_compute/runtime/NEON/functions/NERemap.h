/*
 * Copyright (c) 2016-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_NEREMAP_H
#define ARM_COMPUTE_NEREMAP_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to execute remap. This function calls the following kernels:
 *
 * -# @ref NERemapKernel
 */
class NERemap : public INESimpleFunctionNoBorder
{
public:
    /** Initialise the function's sources, destination, interpolation policy and border mode.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0   |src1   |src2   |dst    |
     * |:------|:------|:------|:------|
     * |U8     |F32    |F32    |U 8    |
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]      map_x                 Map for X coordinates. Data type supported: F32.
     * @param[in]      map_y                 Map for Y coordinates. Data type supported: F32.
     * @param[out]     output                Output tensor. Data type supported: U8.
     * @param[in]      policy                Interpolation policy to use. Only NEAREST and BILINEAR are supported.
     * @param[in]      border_mode           Border mode to use on the input tensor.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT. Defaults to 0.
     *
     * @deprecated This function is deprecated and is intended to be removed in 22.02 release
     *
     */
    ARM_COMPUTE_DEPRECATED_REL(21.08)
    void configure(ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output,
                   InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Initialise the function's sources, destination, interpolation policy and border mode.
     *
     * @param[in, out] input                 Source tensor. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]      map_x                 Map for X coordinates. Data type supported: F32.
     * @param[in]      map_y                 Map for Y coordinates. Data type supported: F32.
     * @param[out]     output                Output tensor. Data type supported: U8.
     * @param[in]      policy                Interpolation policy to use. Only NEAREST and BILINEAR are supported.
     * @param[in]      border_mode           Border mode to use on the input tensor.
     * @param[in]      constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    void configure(ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output,
                   InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value);
};
}
#endif /*ARM_COMPUTE_NEREMAP_H */
