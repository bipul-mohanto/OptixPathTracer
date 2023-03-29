
#include "Util.h"
#include <sutil/vec_math.h>

typedef float Float;
typedef float3 Vector3f;
typedef float2 Point2f;
typedef float3 Spectrum;

static const float ShadowEpsilon = 0.0001f;
static const float Pi = 3.14159265358979323846;
static const float InvPi = 0.31830988618379067154;
static const float Inv2Pi = 0.15915494309189533577;
static const float Inv4Pi = 0.07957747154594766788;
static const float PiOver2 = 1.57079632679489661923;
static const float PiOver4 = 0.78539816339744830961;
static const float Sqrt2 = 1.41421356237309504880;

static const float FloatOneMinusEpsilon = 0x1.fffffep-1;
static const Float OneMinusEpsilon = FloatOneMinusEpsilon;

CUDA_CALLABLE CUDA_INLINE void swap(float& a, float& b) {
	float temp = a;
	a = b;
	b = temp;
}

CUDA_CALLABLE CUDA_INLINE float max(const float& a, const float& b)
{
	return a > b ? a : b;
}

CUDA_CALLABLE CUDA_INLINE float min(const float& a, const float& b)
{
    return a < b ? a : b;
}

CUDA_CALLABLE CUDA_INLINE float sqr(const float& a)
{
	return a * a;
}

CUDA_CALLABLE CUDA_INLINE float3 sqrt(const float3& a)
{
	return make_float3(sqrt(a.x),sqrt(a.y),sqrt(a.z));
}

CUDA_CALLABLE CUDA_INLINE float Lerp(float t, float v1, float v2) { return (1 - t) * v1 + t * v2; }

CUDA_CALLABLE CUDA_INLINE float3 Lerp(float t, const float3& s1, const float3& s2) {
	return (1 - t) * s1 + t * s2;
}

CUDA_CALLABLE CUDA_INLINE float Clamp(const float& val, const float& low, const float& high) {
	return clamp(val, low, high);
}

CUDA_CALLABLE CUDA_INLINE float AbsCosTheta(const float3& w) { return abs(w.z); }
CUDA_CALLABLE CUDA_INLINE float AbsDot(const float3& v1, const float3& v2) {   
    return abs(dot(v1, v2));
}

CUDA_CALLABLE CUDA_INLINE Float CosTheta(const Vector3f& w) { return w.z; }
CUDA_CALLABLE CUDA_INLINE Float Cos2Theta(const Vector3f& w) { return w.z * w.z; }
CUDA_CALLABLE CUDA_INLINE Float Sin2Theta(const Vector3f& w) {
    return max((Float)0, (Float)1 - Cos2Theta(w));
}

CUDA_CALLABLE CUDA_INLINE Float SinTheta(const Vector3f& w) { return std::sqrt(Sin2Theta(w)); }

CUDA_CALLABLE CUDA_INLINE Float TanTheta(const Vector3f& w) { return SinTheta(w) / CosTheta(w); }

CUDA_CALLABLE CUDA_INLINE Float Tan2Theta(const Vector3f& w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

CUDA_CALLABLE CUDA_INLINE float CosPhi(const float3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}

CUDA_CALLABLE CUDA_INLINE float SinPhi(const float3& w) {
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

CUDA_CALLABLE CUDA_INLINE float Cos2Phi(const float3& w) { return CosPhi(w) * CosPhi(w); }

CUDA_CALLABLE CUDA_INLINE float Sin2Phi(const float3& w) { return SinPhi(w) * SinPhi(w); }

CUDA_CALLABLE CUDA_INLINE float3 SphericalDirection(float sinTheta, float cosTheta, float phi) {
	return make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

CUDA_CALLABLE CUDA_INLINE bool SameHemisphere(const float3& w, const float3& wp) {
	return w.z * wp.z > 0;
}

CUDA_CALLABLE CUDA_INLINE float3 Faceforward(const float3& v, const float3& v2) {
    return (dot(v, v2) < 0.f) ? -v : v;
}

CUDA_CALLABLE CUDA_INLINE bool Refract(const Vector3f& wi, const float3& n, Float eta, Vector3f* wt) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    Float cosThetaI = dot(n, wi);
    Float sin2ThetaI = max(Float(0), Float(1 - cosThetaI * cosThetaI));
    Float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    Float cosThetaT = std::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vector3f(n);
    return true;
}

CUDA_CALLABLE CUDA_INLINE Point2f ConcentricSampleDisk(const Point2f& u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - make_float2(1.f, 1.f);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return make_float2(0.f, 0.f);

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * make_float2(cos(theta), std::sin(theta));
}

CUDA_CALLABLE CUDA_INLINE Vector3f CosineSampleHemisphere(const Point2f& u) {
    Point2f d = ConcentricSampleDisk(u);
    Float z = sqrt(max((Float)0, 1 - d.x * d.x - d.y * d.y));
    return make_float3(d.x, d.y, z);
}

CUDA_CALLABLE CUDA_INLINE float Luminance(const float3& c) {
    const Float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
    return YWeight[0] * c.x + YWeight[1] * c.y + YWeight[2] * c.z;
}

CUDA_CALLABLE CUDA_INLINE bool IsBlack(const float3& c) {
    if (c.x != 0) return false;
    if (c.y != 0) return false;
    if (c.z != 0) return false;
    return true;
}

// BxDF Utility Functions
CUDA_CALLABLE CUDA_INLINE float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = Clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = std::abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(max(0.f, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    float cosThetaT = sqrt(max(0.f, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}