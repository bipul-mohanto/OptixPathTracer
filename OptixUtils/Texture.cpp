#include "Texture.h"

#include "Maths.h"

// Implement a simple Gaussian 3x3 filter with sigma = 0.5
// Needed for the CDF generation of the importance sampled HDR environment texture light.
static float gaussianFilter(const float* rgba, unsigned int width, unsigned int height, unsigned int x, unsigned int y)
{
	// Lookup is repeated in x and clamped to edge in y.
	unsigned int left = (0 < x) ? x - 1 : width - 1; // repeat
	unsigned int right = (x < width - 1) ? x + 1 : 0;         // repeat
	unsigned int bottom = (0 < y) ? y - 1 : y;         // clamp
	unsigned int top = (y < height - 1) ? y + 1 : y;         // clamp

	// Center
	const float* p = rgba + (width * y + x) * 4;
	float intensity = (p[0] + p[1] + p[2]) * 0.619347f;

	// 4-neighbours
	p = rgba + (width * bottom + x) * 4;
	float f = p[0] + p[1] + p[2];
	p = rgba + (width * y + left) * 4;
	f += p[0] + p[1] + p[2];
	p = rgba + (width * y + right) * 4;
	f += p[0] + p[1] + p[2];
	p = rgba + (width * top + x) * 4;
	f += p[0] + p[1] + p[2];
	intensity += f * 0.0838195f;

	// 8-neighbours corners
	p = rgba + (width * bottom + left) * 4;
	f = p[0] + p[1] + p[2];
	p = rgba + (width * bottom + right) * 4;
	f += p[0] + p[1] + p[2];
	p = rgba + (width * top + left) * 4;
	f += p[0] + p[1] + p[2];
	p = rgba + (width * top + right) * 4;
	f += p[0] + p[1] + p[2];
	intensity += f * 0.0113437f;

	return intensity / 3.0f;
}

// Create cumulative distribution function for importance sampling of spherical environment lights.
// This is a textbook implementation for the CDF generation of a spherical HDR environment.
// See "Physically Based Rendering" v2, chapter 14.6.5 on Infinite Area Lights.
bool Texture::calculateCDF(OptixDeviceContext& context)
{
    if (m_texels.empty() || (m_texels.size() != m_width * m_height * 4))
    {
        return false;
    }

    const float* rgba = m_texels.data();

    // The original data needs to be retained to calculate the PDF.
    float* funcU = new float[m_width * m_height];
    float* funcV = new float[m_height + 1];

    float sum = 0.0f;
    // First generate the function data.
    for (unsigned int y = 0; y < m_height; ++y)
    {
        // Scale distibution by the sine to get the sampling uniform. (Avoid sampling more values near the poles.)
        // See Physically Based Rendering v2, chapter 14.6.5 on Infinite Area Lights, page 728.
        float sinTheta = float(sin(M_PI * (double(y) + 0.5) / double(m_height))); // Make this as accurate as possible.

        for (unsigned int x = 0; x < m_width; ++x)
        {
            // Filter to keep the piecewise linear function intact for samples with zero value next to non-zero values.
            const float value = gaussianFilter(rgba, m_width, m_height, x, y);
            funcU[y * m_width + x] = value * sinTheta;

            // Compute integral over the actual function.
            const float* p = rgba + (y * m_width + x) * 4;
            const float intensity = (p[0] + p[1] + p[2]) / 3.0f;
            sum += intensity * sinTheta;
        }
    }

    // This integral is used inside the light sampling function (see sysEnvironmentIntegral).
    m_integral = sum * 2.0f * M_PIf * M_PIf / float(m_width * m_height);

    // Now generate the CDF data.
    // Normalized 1D distributions in the rows of the 2D buffer, and the marginal CDF in the 1D buffer.
    // Include the starting 0.0f and the ending 1.0f to avoid special cases during the continuous sampling.
    float* cdfU = new float[(m_width + 1) * m_height];
    float* cdfV = new float[m_height + 1];

    for (unsigned int y = 0; y < m_height; ++y)
    {
        unsigned int row = y * (m_width + 1); // Watch the stride!
        cdfU[row + 0] = 0.0f; // CDF starts at 0.0f.

        for (unsigned int x = 1; x <= m_width; ++x)
        {
            unsigned int i = row + x;
            cdfU[i] = cdfU[i - 1] + funcU[y * m_width + x - 1]; // Attention, funcU is only m_width wide! 
        }

        const float integral = cdfU[row + m_width]; // The integral over this row is in the last element.
        funcV[y] = integral;                        // Store this as function values of the marginal CDF.

        if (integral != 0.0f)
        {
            for (unsigned int x = 1; x <= m_width; ++x)
            {
                cdfU[row + x] /= integral;
            }
        }
        else // All texels were black in this row. Generate an equal distribution.
        {
            for (unsigned int x = 1; x <= m_width; ++x)
            {
                cdfU[row + x] = float(x) / float(m_width);
            }
        }
    }

    // Now do the same thing with the marginal CDF.
    cdfV[0] = 0.0f; // CDF starts at 0.0f.
    for (unsigned int y = 1; y <= m_height; ++y)
    {
        cdfV[y] = cdfV[y - 1] + funcV[y - 1];
    }

    const float integral = cdfV[m_height]; // The integral over this marginal CDF is in the last element.
    funcV[m_height] = integral;            // For completeness, actually unused.

    if (integral != 0.0f)
    {
        for (unsigned int y = 1; y <= m_height; ++y)
        {
            cdfV[y] /= integral;
        }
    }
    else // All texels were black in the whole image. Seriously? :-) Generate an equal distribution.
    {
        for (unsigned int y = 1; y <= m_height; ++y)
        {
            cdfV[y] = float(y) / float(m_height);
        }
    }

    m_texture.Create<float4>(m_width, m_height, (void*)rgba, m_width * m_height * sizeof(float4));
        
    /*m_BufferCDF_U

    // Upload the CDFs into OptiX buffers.
    m_bufferCDF_U = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, m_width + 1, m_height);

    void* buf = m_bufferCDF_U->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
    memcpy(buf, cdfU, (m_width + 1) * m_height * sizeof(float));
    m_bufferCDF_U->unmap();

    m_bufferCDF_V = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, m_height + 1);

    buf = m_bufferCDF_V->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
    memcpy(buf, cdfV, (m_height + 1) * sizeof(float));
    m_bufferCDF_V->unmap();

    delete[] cdfV;
    delete[] cdfU;

    delete[] funcV;
    delete[] funcU;

    m_texels.clear(); // The original float data is not needed anymore.

    return true;*/
}