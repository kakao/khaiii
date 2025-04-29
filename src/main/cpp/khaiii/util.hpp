/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */

#ifndef SRC_MAIN_CPP_KHAIII_UTIL_HPP_
#define SRC_MAIN_CPP_KHAIII_UTIL_HPP_

//////////////
// includes //
//////////////
#ifdef _WIN32
#include <shlwapi.h>
#else
#include <sys/stat.h>
#endif    // _WIN32

#include <sstream>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <cassert>
#include <memory>
#include <iomanip>
#include <map>

#include "boost/locale/encoding_utf.hpp"

#pragma once

namespace khaiii {
    ///////////////
    // functions //
    ///////////////
    /**
     * whether is space or not
     * @param  chr  character
     * @return  true if character is space
     */
    inline bool is_space(wchar_t chr) {
        static std::wstring space(L" \t\v\r\n\u3000");
        return space.find(chr) != std::wstring::npos;
    }

    /**
     * convert UTF-8 string to wstring
     * @param str  UTF-8 string
     * @return  wstring
     */
    inline std::wstring utf8_to_wstr(const char* str) {
        assert(str);
        return boost::locale::conv::utf_to_utf<wchar_t>(str, str + ::strlen(str));
    }

    /**
     * convert wstring to UTF-8 string
     * @param  wstr  wstring
     * @return  UTF-8 string
     */
    inline std::string wstr_to_utf8(const wchar_t* wstr) {
        assert(wstr);
        return boost::locale::conv::utf_to_utf<char>(wstr, wstr + ::wcslen(wstr));
    }

    /**
     * string splitter
     * @param  str  string to split
     * @param  deilm  delimiter char
     * @return  list of splitted strings
     */
    inline std::vector<std::string> split(const char* str, char delim) {
        std::stringstream sss(str);
        std::vector<std::string> elems;
        for (std::string item; std::getline(sss, item, delim);) {
            elems.emplace_back(std::move(item));
        }
        return elems;
    }

    inline std::string const_char_to_bytestring(const char* str) {
        if (str == nullptr) {
            return "nullptr";
        }
        std::stringstream hex_stream;
        std::map<int, std::string> hex_map = {
            { 0x07, R"(\a)" },
            { 0x08, R"(\b)" },
            { 0x1b, R"(\e)" },
            { 0x0c, R"(\f)" },
            { 0x0a, R"(\n)" },
            { 0x0d, R"(\r)" },
            { 0x09, R"(\t)" },
            { 0x0b, R"(\v)" },
        };

        for (int i = 0; str[i] != '\0'; i++) {
            unsigned char byte = static_cast<unsigned char>(str[i]);
            if (static_cast<int>(' ') <= byte && byte <= static_cast<int>('~')) {
                hex_stream << str[i];
            }
            else if (hex_map.find(byte) != hex_map.end()) {
                hex_stream << hex_map.at(byte);
            }
            else {
                hex_stream << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<int>(byte);
            }
        }

        return hex_stream.str();
    }

    /**
     * whether file (or directory) exists or not
     * @param  path  path
     * @return  true if exists
     */
    inline bool file_exists(const char* path) {
        assert(path);
#ifdef _WIN32
        return PathFileExistsA(path);
#else
        struct stat st;
        return stat(path, &st) == 0;
#endif    // _WIN32
    }
} // namespace khaiii

#endif    // SRC_MAIN_CPP_KHAIII_UTIL_HPP_
