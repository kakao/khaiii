/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_
#define SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_

#ifdef _WIN32

/** We suggest the environment as Windows. */
#include <Windows.h>

#define khaiii_when_Windows(...)	__VA_ARGS__
#define khaiii_when_POSIX(...)

#else

/** We suggest the environment as POSIX. */

#include <fcntl.h>
#include <sys/mman.h>

#define khaiii_when_Windows(...)
#define khaiii_when_POSIX(...)		__VA_ARGS__


#endif

/** @brief File descriptor */
#define khaiii_fd \
	khaiii_when_Windows(HANDLE) \
	khaiii_when_POSIX(int)

typedef khaiii_when_Windows(union) 
	khaiii_when_POSIX(struct)

	khaiii_fmap 
{
	khaiii_fd file, map;
} khaiii_fmap;

/**
 * @macro khaiii_rdopen
 *
 * @brief 
 * Open a file with read-only mode and configure for mmap. \n
 * in a permission of `rw-rw----`
 * */
#define khaiii_fmapopen(fmap, path) \
{ \
	(fmap).file = :: \
			khaiii_when_POSIX(open(path, O_RDONLY, 0660)) \
			khaiii_when_Windows( \
					CreateFileA( \
						path \
						, GENERIC_READ \
						, FILE_SHARE_READ \
						, NULL \
						, OPEN_EXISTING \
						, FILE_ATTRIBUTE_NORMAL \
						, NULL  \
						) \
						); \
	khaiii_when_Windows( \
			if((fmap).file != INVALID_HANDLE_VALUE) \
				(fmap).map = \
				::CreateFileMapping( \
						(fmap).file \
						, NULL \
						, PAGE_READWRITE \
						, 0, 0, NULL \
						) \
			) \
}

/** @brief The file descriptor is non-good */
#define khaiii_fd_BAD \
	khaiii_when_Windows(INVALID_HANDLE_VALUE) \
	khaiii_when_POSIX(-1)

/** @brief Check if The fmap is non-good */
#define khaiii_fmapisbad(fmap) \
	((fmap).file == khaiii_fd_BAD \
	 khaiii_when_Windows( \
		 || (fmap).map == NULL  \
		 || (fmap).map == INVALID_HANDLE_VALUE \
		 ))

/** @brief Close a file descriptor */
#define khaiii_fdclose :: \
	khaiii_when_Windows(CloseHandle) \
	khaiii_when_POSIX(close)

/** @brief Close a fmap */
#define khaiii_fmapclose(fmap) \
{ \
	if((fmap).file != khaiii_fd_BAD) \
	khaiii_fdclose((fmap).file); \
	\
	khaiii_when_Windows( \
			if((fmap).map && (fmap).map != INVALID_HANDLE_VALUE) \
			khaiii_fdclose((fmap).map); \
			) \
}

/** @brief mmap read-only */
#define khaiii_rmmap(fmap, addr, len, off) :: \
	khaiii_when_POSIX( \
			mmap( \
				addr \
				, len \
				, PROT_READ \
				, MAP_SHARED \
				, (fmap).map \
				, off \
				) \
			) \
	khaiii_when_Windows( \
			MapViewOfFile( \
				(fmap).map \
				, FILE_MAP_READ \
				, (DWORD)(off >> 32) \
				, (DWORD)(off & 0xFFFFFFFF) \
				, len \
				) \
			)

/** @brief munmap */
#define khaiii_unmap(addr, len) :: \
	khaiii_when_Windows((UnmapViewOfFile(addr) ? -1 : 0)) \
	khaiii_when_POSIX(munmap(addr, len))

/** @brief value when khaiii_rmmap failed. */
#define khaiii_map_FAILED  \
	khaiii_when_Windows(0) \
	khaiii_when_POSIX(MAP_FAILED)

//////////////
// includes //
//////////////

#include <unistd.h>
#include <fstream>
#include <string>
#include <cassert>

#include "fmt/format.h"
#include "khaiii/KhaiiiApi.hpp"


namespace khaiii {


/**
 * memory mapped file
 */
template<typename T>
class MemMapFile {
 public:

    /**
     * dtor
     */
    virtual ~MemMapFile() {
        close();
    }

    /**
     * open memory mapped file
     * @param  path  path
     */
    void open(const char* path) {
        close();
	khaiii_fmapopen(_fmap, path);
	if (khaiii_fmapisbad(_fmap)) 
		throw Except(fmt::format(
					"fail to open file: {}"
					, path)
				);

        std::ifstream fin(path, std::ifstream::ate | std::ifstream::binary);
        _byte_len = fin.tellg();
        if (_byte_len == -1)
		throw Except(fmt::format("fail to get size of file: {}", path));

        assert(_byte_len % sizeof(T) == 0);
        _data = reinterpret_cast<const T*>(
			khaiii_rmmap(
				_fmap
				, 0 /** Not handled this value for windows when non-0. */
				, _byte_len
				, 0
				)
			);

        if (_data == khaiii_map_FAILED) {
            throw Except(fmt::format("fail to map file to memory: {}", path));
        }
        _path = path;
    }

    /**
     * close memory mapped file
     */
    void close() {
        if (_data) {
            if (khaiii_unmap(const_cast<T*>(_data), _byte_len) == -1) {
                throw Except(fmt::format("fail to close memory mapped file: {}", _path));
            }
        }

	khaiii_fmapclose(_fmap);
        _path = "";
        _data = nullptr;
        _byte_len = -1;

    }

	

    /**
     * get pointer of data
     * @return  start address of data
     */
    const T* data() const {
        assert(_data != nullptr && _byte_len >= sizeof(T));
        return _data;
    }

    /**
     * get data size
     * @return  number of data elements (not byte length)
     */
    int size() const {
        assert(_data != nullptr && _byte_len >= sizeof(T));
        return _byte_len / sizeof(T);
    }

 private:
    std::string _path;    ///< file path
    const T* _data = nullptr;    ///< memory data
    int _byte_len = -1;    /* < byte length */
    khaiii_fmap _fmap = { 0, 0 };
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_
