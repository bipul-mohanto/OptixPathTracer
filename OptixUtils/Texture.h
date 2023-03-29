
#include <sutil/sutil.h>
#include <optix.h>

#include "CUDABuffer.h"
#include "CUDATexture.h"

#include <vector>

class Texture 
{		
	int m_width;
	int m_height;

	std::vector<float> m_texels;
	float m_integral;
	CUDABuffer m_bufferCDF_U;
	CUDABuffer m_bufferCDF_V;

	CUDATexture m_texture;

	// Special functions for spherical environment textures.
	void createEnvironment();
	bool calculateCDF(OptixDeviceContext& context);
	float getIntegral() const;	
};