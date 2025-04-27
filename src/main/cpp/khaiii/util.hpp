/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_UTIL_HPP_
#define SRC_MAIN_CPP_KHAIII_UTIL_HPP_


//////////////
// includes //
//////////////
#include <sys/stat.h>

#include <sstream>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <cassert>
#include <memory>

#include "boost/locale/encoding_utf.hpp"


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
    for (std::string item; std::getline(sss, item, delim); ) {
        elems.emplace_back(std::move(item));
    }
    return elems;
}


/**
 * whether file (or directory) exists or not
 * @param  path  path
 * @return  true if exists
 */
inline bool file_exists(const char* path) {
	assert(path);
    struct stat st;
    return stat(path, &st) == 0;
}


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_UTIL_HPP_
