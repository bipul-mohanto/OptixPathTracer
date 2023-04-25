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

#include <optix_device.h>
#include <random.h>

#include <sutil/vec_math.h>

#include "LaunchParams.h"
  
/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" {
     __constant__ LaunchParams launchParams;
}



static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------
  
extern "C" __global__ void __closesthit__radiance()
{ 
    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int   primID = optixGetPrimitiveIndex();
    const int3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const float3& A = sbtData.vertex[index.x];
    const float3& B = sbtData.vertex[index.y];
    const float3& C = sbtData.vertex[index.z];
    float3 Ng = cross(B - A, C - A);
    float3 Ns = (sbtData.normal)
        ? ((1.f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
        : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const float3 rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    float3 diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
        const float2 tc = (1.f - u - v)* sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= make_float3(fromTexture);
    }

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const float3 surfPos
        = (1.f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z];
    const float3 lightPos = make_float3(-907.108f, 2205.875f, -400.0267f);
    const float3 lightDir = lightPos - surfPos;

    // trace shadow ray:
    float3 lightVisibility = make_float3(0.f);
    // the values we store the PRD pointer in:
    unsigned int u0, u1;
    packPointer(&lightVisibility, u0, u1);
    optixTrace(launchParams.traversable,
        surfPos + 1e-3f * Ng,
        lightDir,
        1e-3f,      // tmin
        1.f - 1e-3f,  // tmax
        0.0f,       // rayTime
        OptixVisibilityMask(255),
        // For shadow rays: skip any/closest hit shaders and terminate on first
        // intersection with anything. The miss shader is used to mark if the
        // light was visible.
        OPTIX_RAY_FLAG_DISABLE_ANYHIT
        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        SHADOW_RAY_TYPE,            // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SHADOW_RAY_TYPE,            // missSBTIndex 
        u0, u1);

    // ------------------------------------------------------------------
    // final shading: a bit of ambient, a bit of directional ambient,
    // and directional component based on shadowing
    // ------------------------------------------------------------------
    const float cosDN
        = 0.1f
        + .8f * fabsf(dot(rayDir, Ns));

    float3& prd = *(float3*)getPRD<float3>();
    prd = (.1f + (.2f + .8f * lightVisibility) * cosDN) * diffuseColor;
}
  
extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __miss__radiance()
{
    *getPRD<float3>() = make_float3(1.0f);
}

extern "C" __global__ void __anyhit__shadow()
{ 
    *getPRD<float3>() = make_float3(0.0f);
    optixTerminateRay();
}

extern "C" __global__ void __closesthit__shadow()
{ /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __miss__shadow()
{ 
    *getPRD<float3>() = make_float3(1.0f);
}

  




//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = launchParams.camera;

    float3 pixelColorPRD = make_float3(0.f,0.f,0.f);

    // the values we store the PRD pointer in:
    unsigned int u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const float2 screen(make_float2(ix + .5f, iy + .5f)
        / make_float2(launchParams.frame.fbSize));

    // generate ray direction
    float3 rayDir = normalize(camera.direction
        + (screen.x - 0.5f) * camera.horizontal
        + (screen.y - 0.5f) * camera.vertical);

    optixTrace(launchParams.traversable,
        camera.position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const unsigned int rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const unsigned int fbIndex = ix+iy* launchParams.frame.fbSize.x;
    launchParams.frame.colorBuffer[fbIndex] = rgba;
}

