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

#include <rendering/subpasses/forward_subpass.h>
#include <rendering/postprocessing_renderpass.h>
#include <platform/platform.h>
#include "acl_pipeline.h"

#include "style_transfer_post_processing.h"

constexpr uint32_t OFFSCREEN_IMAGE_WIDTH = 256;
constexpr uint32_t OFFSCREEN_IMAGE_HEIGHT = 512;

style_transfer_post_processing::style_transfer_post_processing()
{
	add_device_extension(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);

	add_device_extension(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
	add_device_extension(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
	add_device_extension(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
	add_device_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	add_instance_extension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
	add_instance_extension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	add_device_extension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	add_device_extension(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
	add_device_extension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
}

bool style_transfer_post_processing::prepare(vkb::Platform &platform)
{
	if (!VulkanSample::prepare(platform))
	{
		return false;
	}

	set_name("Style Transfer");

	load_scene("scenes/sponza/Sponza01.gltf");

	auto &camera_node = vkb::add_free_camera(*scene, "main_camera", get_render_context().get_surface_extent());
	camera = dynamic_cast<vkb::sg::PerspectiveCamera *>(&camera_node.get_component<vkb::sg::Camera>());

	vkb::ShaderSource scene_vs("base.vert");
	vkb::ShaderSource scene_fs("base.frag");
	auto scene_subpass = std::make_unique<vkb::ForwardSubpass>(get_render_context(), std::move(scene_vs), std::move(scene_fs), *scene, *camera);
	scene_pipeline = std::make_unique<vkb::RenderPipeline>();
	scene_pipeline->add_subpass(std::move(scene_subpass));

	vkb::ShaderSource postprocessing_vs("postprocessing/postprocessing.vert");
	final_pipeline = std::make_unique<vkb::PostProcessingPipeline>(get_render_context(), std::move(postprocessing_vs));
	final_pipeline->add_pass().add_subpass(vkb::ShaderSource("postprocessing/simple.frag"));

	stats->request_stats({vkb::StatIndex::frame_times});
	gui = std::make_unique<vkb::Gui>(*this, platform.get_window(), stats.get());

	VkExtent3D offscreen_image_extent{static_cast<uint32_t>(OFFSCREEN_IMAGE_WIDTH),
									  static_cast<uint32_t>(OFFSCREEN_IMAGE_HEIGHT),
									  1};
	nn_pipeline = std::make_unique<ACLPipeline>(OFFSCREEN_IMAGE_WIDTH, OFFSCREEN_IMAGE_HEIGHT, 4);

	for(uint32_t i = 0; i < render_context->get_swapchain().get_images().size(); i++)
	{
		auto offscreen_render_target = create_offscreen_render_target(offscreen_image_extent);
		offscreen_render_targets.push_back(std::move(offscreen_render_target));
	}

	return true;
}

void style_transfer_post_processing::prepare_render_context()
{
	get_render_context().prepare(1, std::bind(&style_transfer_post_processing::create_render_target, this, std::placeholders::_1));
}

std::unique_ptr<vkb::RenderTarget> style_transfer_post_processing::create_render_target(vkb::core::Image &&swapchain_image)
{
	std::vector<vkb::core::Image> images;

	i_swapchain = 0;
	images.push_back(std::move(swapchain_image));

	return std::make_unique<vkb::RenderTarget>(std::move(images));
}

std::unique_ptr<vkb::RenderTarget> style_transfer_post_processing::create_offscreen_render_target(const VkExtent3D& extent)
{
	// Memory for the attachments are allocated in a specific way to support AHardwareBuffer export.

	auto &device = get_device();
	auto depth_format = vkb::get_suitable_depth_format(device.get_gpu().get_handle());

	vkb::core::Image depth_image{device,
								 extent,
								 depth_format,
								 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
								 VMA_MEMORY_USAGE_GPU_ONLY};

	VkExternalMemoryImageCreateInfo external_memory_image_create_info = {};
	external_memory_image_create_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
	external_memory_image_create_info.pNext       = nullptr,
	external_memory_image_create_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

	VkImageCreateInfo image_create_info = {};
	image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_create_info.pNext             = &external_memory_image_create_info;
	image_create_info.imageType         = VK_IMAGE_TYPE_2D;
	image_create_info.format            = VK_FORMAT_R8G8B8A8_UNORM;
	image_create_info.mipLevels         = 1;
	image_create_info.arrayLayers       = 1;
	image_create_info.samples           = VK_SAMPLE_COUNT_1_BIT;
	image_create_info.tiling            = VK_IMAGE_TILING_LINEAR;
	image_create_info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	image_create_info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
	image_create_info.extent            = extent;
	image_create_info.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	VkImage offscreen_image_handle;
	auto result = vkCreateImage(device.get_handle(), &image_create_info, nullptr, &offscreen_image_handle);
	if(result != VK_SUCCESS)
	{
		throw std::runtime_error("Cannot create exported image.");
	}

	VkMemoryDedicatedAllocateInfo dedicated_allocate_info;
	dedicated_allocate_info.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
	dedicated_allocate_info.pNext  = nullptr;
	dedicated_allocate_info.buffer = VK_NULL_HANDLE;
	dedicated_allocate_info.image  = offscreen_image_handle;

	VkMemoryRequirements memory_requirements{};
	vkGetImageMemoryRequirements(device.get_handle(), offscreen_image_handle, &memory_requirements);

	VkExportMemoryAllocateInfo export_memory_allocate_Info;
	export_memory_allocate_Info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
	export_memory_allocate_Info.pNext       = &dedicated_allocate_info;
	export_memory_allocate_Info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

	VkMemoryAllocateInfo memory_allocate_info = {};
	memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memory_allocate_info.pNext                = &export_memory_allocate_Info;
	memory_allocate_info.allocationSize       = 0;
	memory_allocate_info.memoryTypeIndex      = device.get_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

	VkDeviceMemory offscreen_image_memory;
	result = vkAllocateMemory(device.get_handle(), &memory_allocate_info, nullptr, &offscreen_image_memory);
	if(result != VK_SUCCESS)
	{
		throw std::runtime_error("Cannot allocate memory");
	}

	result = vkBindImageMemory(device.get_handle(), offscreen_image_handle, offscreen_image_memory, 0);
	if(result != VK_SUCCESS)
	{
		throw std::runtime_error("Cannot bind memory to image");
	}

	offscreen_memory_allocations.push_back(offscreen_image_memory);
	vkb::core::Image color_image(device, offscreen_image_handle, extent, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	std::vector<vkb::core::Image> images;

	i_offscreen_color = 0;
	images.push_back(std::move(color_image));

	i_offscreen_depth = 1;
	images.push_back(std::move(depth_image));

	return std::make_unique<vkb::RenderTarget>(std::move(images));
}

void style_transfer_post_processing::update(float delta_time)
{
	VulkanSample::update(delta_time);
}

void style_transfer_post_processing::draw(vkb::CommandBuffer &command_buffer, vkb::RenderTarget &render_target)
{
	auto &offscreen_render_target = *offscreen_render_targets[render_context->get_active_frame_index()];
	auto &offscreen_views = offscreen_render_target.get_views();
	auto &offscreen_queue = device->get_suitable_graphics_queue();
	auto &offscreen_command_buffer = render_context->get_active_frame().request_command_buffer(offscreen_queue);
	offscreen_command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		offscreen_command_buffer.image_memory_barrier(offscreen_views.at(i_offscreen_color), memory_barrier);
	}

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		offscreen_command_buffer.image_memory_barrier(offscreen_views.at(i_offscreen_depth), memory_barrier);
	}

	set_viewport_and_scissor(offscreen_command_buffer, offscreen_render_target.get_extent());
	scene_pipeline->draw(offscreen_command_buffer, offscreen_render_target);
	offscreen_command_buffer.end_render_pass();

	offscreen_command_buffer.end();
	offscreen_queue.submit(offscreen_command_buffer, VK_NULL_HANDLE);
	offscreen_queue.wait_idle();

	if(gui_run_postprocessing)
	{
		auto offscreen_image_buffer = get_hardware_buffer_from_image(offscreen_memory_allocations[render_context->get_active_frame_index()]);
		nn_pipeline->run(offscreen_image_buffer, offscreen_views.at(i_offscreen_color).get_image().get_extent());
	}

	auto &views = render_target.get_views();

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		command_buffer.image_memory_barrier(views.at(i_swapchain), memory_barrier);
	}

	final_renderpass(command_buffer, render_target);

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		memory_barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

		command_buffer.image_memory_barrier(views.at(i_swapchain), memory_barrier);
	}
}

void style_transfer_post_processing::final_renderpass(vkb::CommandBuffer &command_buffer, vkb::RenderTarget &render_target)
{
	auto &offscreen_render_target = *offscreen_render_targets[render_context->get_active_frame_index()];
	auto &offscreen_views = offscreen_render_target.get_views();
	vkb::core::SampledImage sampled_image(offscreen_views[i_offscreen_color]);

	glm::vec4 near_far = {camera->get_far_plane(), camera->get_near_plane(), -1.0f, -1.0f};

	auto &postprocessing_pass = final_pipeline->get_pass(0);
	postprocessing_pass.set_uniform_data(near_far);

	auto &postprocessing_subpass = postprocessing_pass.get_subpass(0);
	postprocessing_subpass.bind_sampled_image("color_sampler", std::move(sampled_image));

	final_pipeline->draw(command_buffer, render_target);

	if (gui)
	{
		gui->draw(command_buffer);
	}

	command_buffer.end_render_pass();
}

AHardwareBuffer* style_transfer_post_processing::get_hardware_buffer_from_image(VkDeviceMemory memory)
{
	AHardwareBuffer* hardware_buffer = nullptr;

	VkMemoryGetAndroidHardwareBufferInfoANDROID get_hardware_buffer_info = {};
	get_hardware_buffer_info.sType  = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
	get_hardware_buffer_info.pNext  = nullptr;
	get_hardware_buffer_info.memory = memory;
	auto result = vkGetMemoryAndroidHardwareBufferANDROID(get_device().get_handle(), &get_hardware_buffer_info, &hardware_buffer);
	if(result != VK_SUCCESS)
	{
		throw std::runtime_error("Cannot get AHardwareBuffer from image");
	}

	return hardware_buffer;
}

void style_transfer_post_processing::draw_gui()
{
	gui->show_options_window(
			[this]() {
				ImGui::Checkbox("Enable post-processing", &gui_run_postprocessing);
			},
			1);
}

std::unique_ptr<vkb::VulkanSample> create_style_transfer_post_processing()
{
    return std::make_unique<style_transfer_post_processing>();
}



