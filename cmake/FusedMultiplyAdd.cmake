include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mfma fma_compiles)
if(fma_compiles)
    include(CheckCXXSourceRuns)
    set(test_src
    	"#include <cmath>
    	double fma_wrap(double x, double y, double z) { return fma(x, y, z); }
    	int main() { double a = fma_wrap(1.2, 3.4, 5.6); return 0; }")
    set(CMAKE_REQUIRED_FLAGS -mfma)
    check_cxx_source_runs("${test_src}" fma_runs)
    if(fma_runs)
        message(STATUS "[khaiii] fused multiply add option enabled")
        add_definitions(-mfma)
    else()
        message(WARNING "[khaiii] cpu does not have fused multiply add instruction")
    endif()
else()
    message(WARNING "[khaiii] compiler does not support fused multiply add option")
endif()
