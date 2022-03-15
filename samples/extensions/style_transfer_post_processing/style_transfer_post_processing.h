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

#include <vulkan_sample.h>
#include <rendering/postprocessing_pipeline.h>
#include <scene_graph/components/perspective_camera.h>
#include "acl_pipeline.h"

class style_transfer_post_processing : public vkb::VulkanSample
{
public:
    style_transfer_post_processing();

	virtual ~style_transfer_post_processing() = default;

	virtual bool prepare(vkb::Platform &platform) override;

	virtual void update(float delta_time) override;

	virtual void draw(vkb::CommandBuffer &command_buffer, vkb::RenderTarget &render_target) override;

	virtual void draw_gui() override;

private:
	virtual void prepare_render_context() override;

	// Create main render target which is associated with the swapchain and is used for displaying the result.
	std::unique_ptr<vkb::RenderTarget> create_render_target(vkb::core::Image &&swapchain_image);

	// Create an offscreen target, which is used for rendering the scene and post-processing.
	std::unique_ptr<vkb::RenderTarget> create_offscreen_render_target(const VkExtent3D& extent);

	// This renderpass displays the post-processed offscreen render target.
	void final_renderpass(vkb::CommandBuffer &command_buffer, vkb::RenderTarget &render_target);

	// Helper function to export AHardwareBuffer handle from Vulkan image memory.
	AHardwareBuffer* get_hardware_buffer_from_image(VkDeviceMemory memory);

	// Offscreen render targets for each swapchain image.
	std::vector<std::unique_ptr<vkb::RenderTarget>> offscreen_render_targets;

	// Memory allocations for offscreen render targets. These allocations support AHardwareBuffer export.
	std::vector<VkDeviceMemory> offscreen_memory_allocations;

	// Used to render the scene to the offscreen render target.
	std::unique_ptr<vkb::RenderPipeline> scene_pipeline{};

	// Ued to display the reult onto the screen.
	std::unique_ptr<vkb::PostProcessingPipeline> final_pipeline{};

	// Post processing using a neural network.
	std::unique_ptr<ACLPipeline> nn_pipeline{};

	vkb::sg::PerspectiveCamera *camera{nullptr};

	// Index of swapchain attachment.
	uint32_t i_swapchain{0};

	// Index of offscreen depth attahment.
	uint32_t i_offscreen_depth{0};

	// Index of offscreen color attahment.
	uint32_t i_offscreen_color{0};

	bool gui_run_postprocessing{false};
};

std::unique_ptr<vkb::VulkanSample> create_style_transfer_post_processing();
