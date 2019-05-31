/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_SENTENCE_HPP_
#define SRC_MAIN_CPP_KHAIII_SENTENCE_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"


namespace khaiii {


class Word;


/**
 * sentence data structure
 */
class Sentence {
 public:
    std::vector<std::shared_ptr<Word>> words;    ///< vector of words

    /**
     * ctor
     * @param  raw  raw sentence (UTF-8)
     */
    explicit Sentence(const char* raw = "");

    void organize();    ///< 결과를 구조화합니다.

    inline int morph_cnt() const {
        return _morph_cnt;
    }

    inline const char* raw_str() const {
        return _raw;
    }

    /**
     * get delta from left word boundary to this character
     * @param  wrd_idx  word index
     * @param  chr_idx  character index
     * @return  delta (always less or equal to 0)
     */
    int get_lwb_delta(int wrd_idx, int chr_idx);

    /**
     * get delta from right word boundary to this character
     * @param  wrd_idx  word index
     * @param  chr_idx  character index
     * @return  delta (always more or equal to 0)
     */
    int get_rwb_delta(int wrd_idx, int chr_idx);

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    const char* _raw = "";    ///< raw sentence (UTF-8)
    int _morph_cnt;    ///< total morpheme count
    std::wstring _wraw;    ///< unicode characters
    std::vector<int> _wbegins;    ///< character begin positions for each unicode characters
    std::vector<int> _wends;    ///< character end positions for each unicode characters

    void _tokenize();    ///< tokenize by spaces
    void _characterize();    ///< convert to unicode characters
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_SENTENCE_HPP_
