/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_
#define SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_


//////////////
// includes //
//////////////
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <string>

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
    void open(std::string path) {
        close();
        int fd = ::open(path.c_str(), O_RDONLY, 0660);
        if (fd == -1) throw Except(fmt::format("fail to open file: {}", path));
        std::ifstream fin(path, std::ifstream::ate | std::ifstream::binary);
        _byte_len = fin.tellg();
        if (_byte_len == -1) throw Except(fmt::format("fail to get size of file: {}", path));
        assert(_byte_len % sizeof(T) == 0);
        _data = reinterpret_cast<const T*>(::mmap(0, _byte_len, PROT_READ, MAP_SHARED, fd, 0));
        ::close(fd);
        if (_data == MAP_FAILED) {
            throw Except(fmt::format("fail to map file to memory: {}", path));
        }
        _path = path;
    }

    /**
     * close memory mapped file
     */
    void close() {
        if (_data) {
            if (::munmap(const_cast<T*>(_data), _byte_len) == -1) {
                throw Except(fmt::format("fail to close memory mapped file: {}", _path));
            }
        }
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
    int _byte_len = -1;    ///< byte length
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_MEMMAPFILE_HPP_
