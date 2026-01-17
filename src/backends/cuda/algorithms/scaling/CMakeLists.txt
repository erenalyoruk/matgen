if(MATGEN_ENABLE_CUDA)
  target_sources(
    matgen
    PRIVATE
      "bilinear_cuda.cu"
      "nearest_neighbor_cuda.cu"
      "lanczos_cuda.cu"
      "fft_cuda.cu"
      "wavelet_cuda.cu"
  )
  target_link_libraries(matgen PRIVATE cufft)
  message(STATUS "CUDA FFT backend enabled (cuFFT)")
endif()