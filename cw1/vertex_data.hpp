#pragma once

#include <cstdint>
#include "model.hpp"
#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 

struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;

	std::uint32_t vertexCount;
};


std::vector<ColorizedMesh> create_triangle_mesh( labutils::VulkanContext const&, labutils::Allocator const&, ModelData& data );



