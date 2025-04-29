# TODO
- Test if it works both on Windows & Linux

# Error Fixed
- Now is using char32_t instead of wchar_t
- Using std::locale::classic(), which is repectively using default locale for OS.
- No locale is being generated.

# What you need
- cmake, and compiler (gcc, clang, ...), and linker (make, ninja, ...)

# Dependent 3rd party libraries

## Automatical installation
> Will automatically be imported if possible.  
> is tested on Arch Linux.  
> For renewing them you will need to delete the cache by removing the build directory, whereever you've set. (Normally it's `out` or `build` or `Build`-like)  
- Threads
- cxxopts
- fmt
- Boost
- GTest
- spdlog

## Libraries which Manual Handling required
> In cmake, some libraries are not allowing in-source builds.  
> You will install them manually.
- Eigen3

## Not sure currently
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
