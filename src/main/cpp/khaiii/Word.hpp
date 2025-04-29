/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_WORD_HPP_
#define SRC_MAIN_CPP_KHAIII_WORD_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <vector>

#include "khaiii/khaiii_api.h"
#include "khaiii/Resource.hpp"
#include "khaiii/Restore.hpp"


namespace khaiii {


class Morph;


/**
 * 어절 자료구조
 */
class Word: public khaiii_word_t {
 public:
    const char32_t* wbegin = nullptr;    ///< unicode string begin address
    int wlength = 0;    ///< unicode string length
    std::vector<std::shared_ptr<Morph>> morph_vec;   ///< 어절에 포함된 형태소 배열 (분석 결과)

    std::vector<embedding_t> embeds;    ///< embeddings for each character
    std::vector<uint16_t> char_tags;    ///< tag outs for each character
    std::vector<std::vector<chr_tag_t>> restored;    ///< restored characters and their tags

    /**
     * ctor
     * @param  wbegin  unicode string begin address
     * @param  length  unicode string length
     */
    explicit Word(const char32_t* wbegin = nullptr, int wlength = 0);

    /**
     * set begin position and length in raw string for this word
     * @param  wchars  unicode characters
     * @param  wbegins  begin positions for each unicode characters
     * @param  wends  end positions for each unicode characters
     */
    void set_begin_length(const char32_t* wchars, const std::vector<int> &wbegins,
                          const std::vector<int> &wends);

    /**
     * set embedding for decoding
     * @param  rsc  resource
     */
    void set_embeds(const Resource& rsc);

    /**
     * 하나의 형태소를 추가한다.
     * @param  wlex  유니코드 형태소 문자열
     * @param  tag  품사 태그 번호 (1부터 시작. 0은 오류)
     * @param  begin_idx  시작 인덱스 (유니코드 음절 인덱스)
     * @param  end_idx  끝 인덱스 (유니코드 음절 인덷스)
     */
    void add_morph(const std::u32stringstream& wlex, uint8_t tag, int begin_idx, int end_idx);

    /**
     * API 결과 구조체의 내용을 채운다.
     * @param  wraw  유니코드 원문
     * @param  wbegins  각 음절별 시작 byte 위치
     * @param  wends  각 음절별 끝 byte 위치
     */
    void organize(const char32_t* wraw, const std::vector<int>& wbegins,
                  const std::vector<int>& wends);

    /**
     * 원형복원된 음절들을 바탕으로 형태소를 생성한다.
     */
    void make_morphs();

    std::string str() const;    ///< to string (UTF-8)
    std::u32string wstr() const;    ///< to unicode string
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_WORD_HPP_
