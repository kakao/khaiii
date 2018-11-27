cmake_minimum_required(VERSION 3.10)

# manage 3rd party libraries with "hunter"
if(NOT APPLE)
    set(HUNTER_BUILD_SHARED_LIBS ON)
endif()
include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.34.tar.gz"
        SHA1 "70287b1ffa810ee4e952052a9adff9b4856d0d54"
        LOCAL
)

# versions
file(READ "include/khaiii/khaiii_api.h" KHAIII_VERSION_HEADER)
string(REGEX MATCH "#define[ \t]+KHAIII_VERSION_MAJOR[ \t]+([0-9]+)" _KHAIII_VERSION_MAJOR "${KHAIII_VERSION_HEADER}")
set(KHAIII_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE STRING "major version" FORCE)
string(REGEX MATCH "#define[ \t]+KHAIII_VERSION_MINOR[ \t]+([0-9]+)" _KHAIII_VERSION_MINOR "${KHAIII_VERSION_HEADER}")
set(KHAIII_VERSION_MINOR ${CMAKE_MATCH_1} CACHE STRING "minor version" FORCE)
set(KHAIII_VERSION ${KHAIII_VERSION_MAJOR}.${KHAIII_VERSION_MINOR} CACHE STRING "version" FORCE)

# project
project(khaiii VERSION ${KHAIII_VERSION})

# options
option(FMA "on/off fused multiply add compiler option" ON)
if(FMA)
    include("cmake/FusedMultiplyAdd.cmake")
endif()
option(PROFILER "on/off profiler option" OFF)
if(PROFILER)
    # bin/khaiii는 /tmp/bin_khaiii.prof 파일로, test/khaiii는 /tmp/test_khaiii.prof 파일로 생성됩니다.
    # 출력 파일명을 지정하고 싶을 경우 CPUPROFILE=/path/to/output bin/khanii와 같이 환경변수와 함께 실행합니다.
    # pprof --text bin/khaiii /tmp/bin_khaiii.prof와 같이 실행하면 출력 파일로부터 텍스트 보고서를 생성할 수 있습니다.
    # 참고: https://github.com/gperftools/gperftools
    message(STATUS "[khaiii] profiler option enabled")
    include("cmake/FindGperftools.cmake")
    add_definitions(-DPROFILER)
endif()
option(COVERAGE "on/off code coverage option" OFF)
if(COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    # 코드 커버리지 옵션을 켭니다.
    # lcov --capture --directory . --output-file coverage.info
    # genhtml coverage.info --output-directory coverage.html
    # python -m lcov_cobertura --excludes build.CMakeFiles --excludes ".*.usr.include.*" --excludes ".*..hunter._Base.*" coverage.info
    message(STATUS "[khaiii] coverage option enabled")
    set(CMAKE_BUILD_TYPE Debug)
    include(cmake/CodeCoverage.cmake)
    APPEND_COVERAGE_COMPILER_FLAGS()
endif()

set(CMAKE_DEBUG_POSTFIX "_debug")

# dependent 3rd party libraries
hunter_add_package(Boost)
find_package(Boost CONFIG REQUIRED)
hunter_add_package(cxxopts)
find_package(cxxopts CONFIG REQUIRED)
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)
hunter_add_package(fmt)
find_package(fmt CONFIG REQUIRED)
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)
hunter_add_package(nlohmann_json)
find_package(nlohmann_json CONFIG REQUIRED)
hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DFMT_HEADER_ONLY)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-O3)
    add_definitions(-g)
    add_definitions(-funroll-loops)
endif()

# main objects
add_library(obj_khaiii OBJECT
        src/main/cpp/khaiii/Config.cpp
        src/main/cpp/khaiii/Embed.cpp
        src/main/cpp/khaiii/ErrPatch.cpp
        src/main/cpp/khaiii/KhaiiiImpl.cpp
        src/main/cpp/khaiii/Morph.cpp
        src/main/cpp/khaiii/Preanal.cpp
        src/main/cpp/khaiii/Resource.cpp
        src/main/cpp/khaiii/Restore.cpp
        src/main/cpp/khaiii/Sentence.cpp
        src/main/cpp/khaiii/Tagger.cpp
        src/main/cpp/khaiii/Trie.cpp
        src/main/cpp/khaiii/Word.cpp
        src/main/cpp/khaiii/khaiii_api.cpp
        src/main/cpp/khaiii/khaiii_dev.cpp
        src/main/cpp/khaiii/nn/Conv1d.cpp
        src/main/cpp/khaiii/nn/Linear.cpp
        src/main/cpp/khaiii/nn/tensor.cpp)
set(include_directories
        include
        src/main/cpp
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR})
target_include_directories(obj_khaiii PUBLIC ${include_directories})
target_compile_definitions(obj_khaiii PUBLIC -DPREFIX="${CMAKE_INSTALL_PREFIX}")

# library: libkhaiii.so
add_library(khaiii SHARED $<TARGET_OBJECTS:obj_khaiii>)
set_target_properties(khaiii PROPERTIES
        VERSION ${KHAIII_VERSION}
        SOVERSION ${KHAIII_VERSION_MAJOR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# executable: bin/khaiii
add_executable(bin_khaiii
        $<TARGET_OBJECTS:obj_khaiii>
        src/main/cpp/main.cpp)
set_target_properties(bin_khaiii PROPERTIES
        OUTPUT_NAME khaiii
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_include_directories(bin_khaiii PRIVATE ${include_directories})
target_link_libraries(bin_khaiii PRIVATE
        cxxopts::cxxopts
        ${GPERFTOOLS_PROFILER}
        ${CMAKE_THREAD_LIBS_INIT})

# executable: test_khaiii
add_executable(test_khaiii
        $<TARGET_OBJECTS:obj_khaiii>
        src/test/cpp/test_main.cpp
        src/test/cpp/khaiii/ErrPatchTest.cpp
        src/test/cpp/khaiii/KhaiiiApiTest.cpp
        src/test/cpp/khaiii/KhaiiiDevTest.cpp
        src/test/cpp/khaiii/PreanalTest.cpp)
set_target_properties(test_khaiii PROPERTIES
        OUTPUT_NAME khaiii
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
target_include_directories(test_khaiii PRIVATE
        src/test/cpp
        ${include_directories})
target_link_libraries(test_khaiii PRIVATE
        GTest::gtest
        cxxopts::cxxopts
        ${GPERFTOOLS_PROFILER}
        ${CMAKE_THREAD_LIBS_INIT})
if(COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(test_khaiii PRIVATE gcov)
endif()

# resource
add_custom_target(resource
        COMMAND make PREFIX=${CMAKE_BINARY_DIR} -C ${CMAKE_SOURCE_DIR}/rsc)
add_custom_target(large_resource
        COMMAND make PREFIX=${CMAKE_BINARY_DIR} MODEL_SIZE=large -C ${CMAKE_SOURCE_DIR}/rsc)

# test
enable_testing()
add_test(NAME test_khaiii COMMAND $<TARGET_FILE:test_khaiii> "--rsc-dir=${CMAKE_BINARY_DIR}/share/khaiii")

# install
install_targets(/lib khaiii)
install_targets(/bin bin_khaiii)
install_files(/include FILES ${CMAKE_SOURCE_DIR}/include/khaiii)
install_files(/share FILES ${CMAKE_BINARY_DIR}/share/khaiii)

# package
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_VENDOR "Kakao Corp.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Kakao Hangul Analyzer III")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_SOURCE_GENERATOR ZIP)
set(CPACK_SOURCE_PACKAGE_FILE_NAME ${CMAKE_PROJECT_NAME}-${KHAIII_VERSION})
set(CPACK_SOURCE_IGNORE_FILES
        "/\\\\..*"
        "/.*build.*/"
        "/train/"
        "__pycache__"
        ".*\\\\.pyc"
        ${CPACK_SOURCE_IGNORE_FILES})
include(CPack)

# package python
set(PKG_PY_DIR ${CMAKE_BINARY_DIR}/package_python)
add_custom_target(package_python
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/README.md ${PKG_PY_DIR}/README.md
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/LICENSE ${PKG_PY_DIR}/LICENSE
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PKG_PY_DIR}/khaiii
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/main/python/khaiii/khaiii.py ${PKG_PY_DIR}/khaiii/khaiii.py
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/src/test/python/test_khaiii ${PKG_PY_DIR}/khaiii/test_khaiii
        COMMAND make package_source
        COMMAND ${CMAKE_COMMAND} -E rename ${CPACK_SOURCE_PACKAGE_FILE_NAME}.zip ${PKG_PY_DIR}/${CPACK_SOURCE_PACKAGE_FILE_NAME}.zip)
configure_file(${CMAKE_SOURCE_DIR}/src/main/python/setup.py.in ${PKG_PY_DIR}/setup.py @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/main/python/MANIFEST.in.in ${PKG_PY_DIR}/MANIFEST.in @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/main/python/khaiii/__init__.py.in ${PKG_PY_DIR}/khaiii/__init__.py @ONLY)
