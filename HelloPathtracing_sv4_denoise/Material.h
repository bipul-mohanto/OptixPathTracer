#pragma once

#include "Maths.h"

#ifndef MATERIAL_HEADER
#define MATERIAL_HEADER

static const int MATERIAL_FLAG_NONE = 0;
static const int MATERIAL_FLAG_SHADOW_CATCHER = 1 << 0; // bm: what is shadow catcher for material?

struct Material
{
	Material()
	{   // bm: parameters to tune the material-brdf 
		//!TODO: better true material interaction
		color = make_float3(1.0f, 0.0f, 0.0f); // bm: no visible effect
		emission = make_float3(1.0f); //0
		absorption = make_float3(1.0); //0

		// when eta is zero the index of refraction will be inferred from the specular component
		eta = 1.4f;

		metallic = 0.5;//0.0f;
		subsurface = 0.0f;
		specular = 1.0;//0.5f;
		roughness = 1.0;//1.0f; //bm:making 0 adding firflies
		specularTint = 1.0f;//0  //bm: this one is changing the scene lighting condition very much
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.0f;
		clearcoat = 0.0f;
		clearcoatGloss = 1.0f;
		transmission = 0.4f;//0// bm: for glasses
		bump = 0.0f; //0
		bumpTile = make_float3(1.0f);		//10; //bm: no visible effect

		flags = 0; // 1 creates ghost effect
	}

	CUDA_CALLABLE __forceinline__ float GetIndexOfRefraction() const
	{
		if (eta == 0.0f)
			return 2.0f / (1.0f - sqrt(0.08f * specular)) - 1.0f;
		else
			return eta;
	}

	float3 emission;
	float3 color;
	float3 absorption;

	float eta;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float transmission;

	//Texture bumpMap;
	float bump;
	float3 bumpTile;

	int flags;
};

#endif // !MATERIAL_HEADER