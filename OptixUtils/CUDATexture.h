
#include "cuda_runtime_api.h"

#include <sutil/Exception.h>

struct CUDATexture {
	cudaTextureObject_t cuda_tex;
	cudaArray_t cuda_array;

    template<typename T>
    void Create(int width, int height, void* data, size_t sizeInBytes){
        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t numComponents = 4;
        int32_t pitch = width * sizeof(T);
        channel_desc = cudaCreateChannelDesc<T>();

        cudaArray_t& pixelArray = cuda_array;
        CUDA_CHECK(cudaMallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
            /* offset */0, 0,
            data,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 0;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cuda_tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    }
};