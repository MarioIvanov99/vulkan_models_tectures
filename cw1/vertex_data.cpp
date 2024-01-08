#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;

std::vector<ColorizedMesh> create_triangle_mesh( labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, ModelData& data )
{
	std::vector<ColorizedMesh> return_mesh;
	for (int j = 0; j < data.meshes.size(); j++) {
		// Vertex data
		std::vector<float> positions;
		std::vector<float> colors;
		std::vector<float> texCoords;

		//If the mesh has no texture, set its colors
		//Else, set its texture coordinates
		//This same if/else is used multiple times throughout this function
		if (data.materials[data.meshes[j].materialIndex].colorTexturePath.compare("") == 0) {
			for (int i = 0; i < data.meshes[j].numberOfVertices; i++) {
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].x);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].y);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].z);
				colors.push_back(data.materials[data.meshes[j].materialIndex].color.x);
				colors.push_back(data.materials[data.meshes[j].materialIndex].color.y);
				colors.push_back(data.materials[data.meshes[j].materialIndex].color.z);
			}
		}
		else {
			for (int i = 0; i < data.meshes[j].numberOfVertices; i++) {
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].x);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].y);
				positions.push_back(data.vertexPositions[data.meshes[j].vertexStartIndex + i].z);
				texCoords.push_back(data.vertexTextureCoords[data.meshes[j].vertexStartIndex + i].x);
				texCoords.push_back(data.vertexTextureCoords[data.meshes[j].vertexStartIndex + i].y);
			}
		}

		//printf("%d", sizeof(colors));

		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			positions.size() * sizeof(float),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);

		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			positions.size() * sizeof(float),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		lut::Buffer vertexColGPU;
		lut::Buffer colStaging;
		lut::Buffer vertexTexGPU;
		lut::Buffer texStaging;

		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

		}

		std::memcpy(posPtr, positions.data(), positions.size() * sizeof(float));
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		if (data.materials[data.meshes[j].materialIndex].colorTexturePath.compare("") == 0) {
			vertexColGPU = lut::create_buffer(
				aAllocator,
				colors.size() * sizeof(float),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			colStaging = lut::create_buffer(
				aAllocator,
				colors.size() * sizeof(float),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* colPtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

			}

			std::memcpy(colPtr, colors.data(), colors.size() * sizeof(float));
			vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);
		}
		else {
			vertexTexGPU = lut::create_buffer(
				aAllocator,
				texCoords.size() * sizeof(float),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			texStaging = lut::create_buffer(
				aAllocator,
				texCoords.size() * sizeof(float),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* texPtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, texStaging.allocation, &texPtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

			}

			std::memcpy(texPtr, texCoords.data(), texCoords.size() * sizeof(float));
			vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);
		}

		lut::Fence uploadComplete = create_fence(aContext);

		lut::CommandPool uploadPool = create_command_pool(aContext);
		VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());

		}

		VkBufferCopy pcopy{};
		pcopy.size = positions.size() * sizeof(float);

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy ccopy{};
		if (data.materials[data.meshes[j].materialIndex].colorTexturePath.compare("") == 0) {
			ccopy.size = colors.size() * sizeof(float);

			vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

			lut::buffer_barrier(uploadCmd,
				vertexColGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);
		}
		else {
			ccopy.size = texCoords.size() * sizeof(float);

			vkCmdCopyBuffer(uploadCmd, texStaging.buffer, vertexTexGPU.buffer, 1, &ccopy);

			lut::buffer_barrier(uploadCmd,
				vertexTexGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);
		}

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Submit transfer commands 
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());

		}


		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());
		}

		if (data.materials[data.meshes[j].materialIndex].colorTexturePath.compare("") == 0) {
			return_mesh.push_back(ColorizedMesh{
			std::move(vertexPosGPU),
			std::move(vertexColGPU),
			(unsigned int)positions.size() / 3 // three floats per position 
				});
		}
		else {
			return_mesh.push_back(ColorizedMesh{
			std::move(vertexPosGPU),
			std::move(vertexTexGPU),
			(unsigned int)positions.size() / 3 // three floats per position 
				});
		}

	}

	return return_mesh;
	
}