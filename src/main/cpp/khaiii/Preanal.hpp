/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_PREANAL_HPP_
#define SRC_MAIN_CPP_KHAIII_PREANAL_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>

#include "spdlog/spdlog.h"

#include "khaiii/MemMapFile.hpp"
#include "khaiii/Trie.hpp"


namespace khaiii {


class Word;


class Preanal {
 public:
    virtual ~Preanal();    ///< dtor

    /**
     * 리소스를 연다.
     * @param  dir  리소스 디렉토리
     */
    void open(const char* dir);

    void close();    ///< 리소스를 닫는다.

    /**
     * 기분석 사전을 적용하여 음절 별로 태깅한다.
     * @param  word  어절
     */
    void apply(std::shared_ptr<Word> word) const;

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    Trie _trie;
    MemMapFile<uint16_t> _val_mmf;    ///< value memory mapping
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_PREANAL_HPP_
