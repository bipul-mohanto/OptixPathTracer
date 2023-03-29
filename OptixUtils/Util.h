

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#define CUDA_INLINE __forceinline__
#else
#define CUDA_CALLABLE
#define CUDA_INLINE inline
#endif