// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cuda_runtime.h>
#include <optix.h>

// our own classes, partly shared between host and device
#include "Model.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Probe.h"
#include "OptixDenoiser.h"

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Matrix.h>


/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
class SampleRenderer
{
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
    public:
    /*! constructor - performs all setup, including initializing
        optix, creates module, pipeline, programs, SBT, etc. */
    SampleRenderer(const Model* model);

    /*! render one frame */
    void render();
    void render(sutil::CUDAOutputBuffer<uint32_t>&);

    /*! resize frame buffer to given resolution */
    void resize(const int2 &newSize);

    /*! download the rendered color buffer */
    void downloadPixels(uint32_t h_pixels[]);

    /*! set camera to render with */
    void setCamera(const sutil::Camera& camera);

    void setProbe(const ProbeData& probe);

    bool denoiserOn = true;

    protected:
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! helper function that initializes optix and checks for errors */
    void initOptix();

    /*! creates and configures a optix device context (in this simple
        example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
        to use. in this simple example, we use a single module from a
        single .cu file, using a single embedded ptx string */
    void createModule();

    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();

    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();

    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel();

    /*! upload textures, and create cuda texture objects for them */
    void createTextures();

    public:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions    pipelineLinkOptions;
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions;
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    CUDABuffer frame_buffer;
    CUDABuffer accum_buffer;

    CUDABuffer normal_buffer;
    CUDABuffer albedo_buffer;
    CUDABuffer color_buffer;


    /*! output of the denoiser pass, in float4 */
    CUDABuffer denoisedBuffer;

    /* the actual final color buffer used for display, in rgba8 */
    CUDABuffer finalColorBuffer;    

    OptiXDenoiser denoiser;

    /*! the camera we are to render with. */
    sutil::Camera lastSetCamera;

    /*! the model we are going to trace rays against */
    const Model* model;
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;
    std::vector<CUDABuffer> indexBuffer;
    std::vector<CUDABuffer> normalBuffer;
    std::vector<CUDABuffer> texcoordBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    /*! @{ one texture object and pixel array per used texture */
    std::vector<cudaArray_t>         textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
    /*! @} */

    CUDAProbeData probeData;
};