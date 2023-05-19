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
 
#include <optix.h>
#include <cuda_runtime.h>

// for this simple example, we have a single ray type
enum { 
    SURFACE_RAY_TYPE = 0, 
    SHADOW_RAY_TYPE = 1, 
    RAY_TYPE_COUNT 
};

struct TriangleMeshSBTData {
    float3  color;
    float3* vertex;
    float3* normal;
    float2* texcoord;
    int3* index;
    bool                hasTexture;
    cudaTextureObject_t texture;
};

struct LaunchParams
  {
    struct {
      unsigned int *colorBuffer;
      int2     fbSize;
    } frame;
    
    struct {
      float3 position;
      float3 direction;
      float3 horizontal;
      float3 vertical;
    } camera;

    OptixTraversableHandle traversable;
  };

struct RayGenData
{
    void* data;
};

struct MissData
{
    void* data;
};

struct HitGroupData
{
    TriangleMeshSBTData data;
};


