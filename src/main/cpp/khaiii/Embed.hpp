/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_EMBED_HPP_
#define SRC_MAIN_CPP_KHAIII_EMBED_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "spdlog/spdlog.h"

#include "khaiii/MemMapFile.hpp"
#include "khaiii/nn/tensor.hpp"


namespace khaiii {


using embedding_t = nn::vector_map_t;
class Config;


class Embed {
 public:
    /**
     * open resource with memory data
     * @param  cfg  config
     * @param  dir  base directory
     */
    void open(const Config& cfg, const char* dir);

    void close();    ///< 리소스를 닫는다.

    /**
     * get embedding vector with character
     * @param  chr  character
     * @return  embedding vector
     */
    const embedding_t& operator[](char32_t chr) const;

    const embedding_t& left_word_bound() const;    ///< left word bound
    const embedding_t& right_word_bound() const;    ///< right word bound
    const embedding_t& left_padding() const;    ///< left padding
    const embedding_t& right_padding() const;    ///< right padding

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    const char32_t* _keys = nullptr;    ///< keys (characters)
    std::vector<embedding_t> _vals;    ///< values (embedding vectors)

    static int _key_cmp(const void* left, const void* right);    ///< key comparator for bsearch

    MemMapFile<char> _embed_mmf;    ///< model embedding memory mapping
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_EMBED_HPP_
