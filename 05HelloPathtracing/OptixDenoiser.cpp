#include "OptixDenoiser.h"

#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <iomanip>

static void context_log_cb_denoiser(uint32_t level, const char* tag, const char* message, void* /*cbdata*/)
{
    if (level < 4)
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

void OptiXDenoiser::init(DenoiseData& data)
{
    
}


void OptiXDenoiser::finish()
{
    if (!m_denoiser)
        return;

    /*const uint64_t frame_byte_size = m_output.width * m_output.height * sizeof(float4);
    CUDA_CHECK(cudaMemcpy(
        m_host_output,
        reinterpret_cast<void*>(m_output.data),
        frame_byte_size,
        cudaMemcpyDeviceToHost
    ));*/

    // Cleanup resources
    optixDenoiserDestroy(m_denoiser);
    optixDeviceContextDestroy(m_context);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_intensity)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));

    m_denoiser = nullptr;
}