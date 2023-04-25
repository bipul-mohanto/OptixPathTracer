OPTIX_add_sample_executable(04HelloRaytracing target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  CUDABuffer.h
  LaunchParams.h
  main.cpp
  SampleRenderer.cpp
  SampleRenderer.h
  deviceProgram.cu    
  Model.cpp
  Model.h
  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )