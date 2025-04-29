This library is being tested on Arch Linux & Windows 11.

# TODO
- Test if Eigen3 Library fetch works on Windows

# Update
- Eigen3 fetching system is now completed. Tested on Arch Linux.

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
- Eigen3
- nlohmann_jaon