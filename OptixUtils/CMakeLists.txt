set(sources
  Texture.h
  Texture.cpp

  CUDABuffer.h
  CUDATexture.h

  Maths.h
  Util.h
)

add_library(OptixUtils OBJECT ${sources})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${sutil_sources})
target_include_directories(OptixUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# Make the list of sources available to the parent directory for installation needs.
set(optix_utils "${sources}" PARENT_SCOPE)

target_link_libraries(OptixUtils LINK_PRIVATE
  ${CUDA_LIBRARIES}
  sutil_7_sdk
  glad
  )