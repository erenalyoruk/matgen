target_sources(
  matgen
  PRIVATE
    "bilinear_seq.c"
    "nearest_neighbor_seq.c"
    "adaptive_seq.c"
    "lanczos_seq.c"
)
if(FFTW3_FOUND)
  message(STATUS "FFTW3 found: enabling FFT sequential backend")
  
  target_sources(
    matgen
    PRIVATE
      "bilinear_seq.c"
      "nearest_neighbor_seq.c"
      "lanczos_seq.c"
      "fft_seq.c"
      "wavelet_seq.c"
  )
  
  target_include_directories(matgen PRIVATE ${FFTW3_INCLUDE_DIRS})
  target_link_libraries(matgen PRIVATE ${FFTW3_LIBRARIES})
  target_compile_definitions(matgen PRIVATE MATGEN_HAS_FFTW3)
else()
  message(WARNING "FFTW3 not found: FFT sequential backend disabled")
  message(STATUS "Install FFTW3 locally: cd && wget http://www.fftw.org/fftw-3.3.10.tar.gz && tar xzf fftw-3.3.10.tar.gz && cd fftw-3.3.10 && ./configure --prefix=$HOME/fftw && make && make install")
  
  target_sources(
    matgen
    PRIVATE
      "bilinear_seq.c"
      "nearest_neighbor_seq.c"
      "lanczos_seq.c"
      "wavelet_seq.c"
  )
endif()
