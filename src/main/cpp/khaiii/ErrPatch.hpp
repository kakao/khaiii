/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_ERRPATCH_HPP_
#define SRC_MAIN_CPP_KHAIII_ERRPATCH_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "khaiii/MemMapFile.hpp"
#include "khaiii/Trie.hpp"


namespace khaiii {


class Sentence;


class ErrPatch {
 public:
    static const char32_t WORD_DELIM_NUM;    ///< 어절 경계를 나타내는 가상 음절
    static const char32_t SENT_DELIM_NUM;    ///< 문장 경계를 나타내는 가상 음절

    virtual ~ErrPatch();    ///< dtor

    /**
     * 리소스를 연다.
     * @param  dir  리소스 디렉토리
     */
    void open(const char* dir);

    void close();    ///< 리소스를 닫는다.

    /**
     * 오분석 패치를 적용한다.
     * @param  sent  문장
     */
    void apply(std::shared_ptr<Sentence> sent) const;

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    Trie _trie;
    MemMapFile<int16_t> _val_mmf;    ///< value memory mapping
    std::vector<const int16_t*> _vals;    ///< actual values

    /**
     * 문장을 Trie 입력에 맞도록 음절과 태그의 비트 조합의 열로 만들고, 출력 위치를 기록한다.
     * @param  sent  문장
     * @param  outputs  출력 위치
     * @return  음절과 태그의 비트 조합한 열
     */
    static std::vector<char32_t> _get_char_tag_mixes(std::shared_ptr<Sentence> sent,
                                                    std::vector<uint16_t*>* outputs);
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_ERRPATCH_HPP_
