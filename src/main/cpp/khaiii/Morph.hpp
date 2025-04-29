/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_MORPH_HPP_
#define SRC_MAIN_CPP_KHAIII_MORPH_HPP_


//////////////
// includes //
//////////////
#include <string>
#include <vector>

#include "khaiii/khaiii_api.h"


namespace khaiii {


/** 품사 태그 */
typedef enum {
     EC,  EF,  EP, ETM, ETN,  IC,  JC, JKB, JKC, JKG,
    JKO, JKQ, JKS, JKV,  JX, MAG, MAJ,  MM, NNB, NNG,
    NNP,  NP,  NR,  SE,  SF,  SH,  SL,  SN,  SO,  SP,
     SS,  SW, SWK,  VA, VCN, VCP,  VV,  VX, XPN,  XR,
    XSA, XSN, XSV,  ZN,  ZV,  ZZ,
    POS_TAG_SIZE
} pos_tag_t;


/**
 * 형태소 자료구조
 */
class Morph: public khaiii_morph_t {
 public:
    std::u32string wlex;    ///< unicode lexical
    const char32_t* wbegin = nullptr;    ///< unicode string begin address
    int wlength = 0;    ///< unicode string length

    Morph(const char32_t* wlex, pos_tag_t tag, const char32_t* wbegin, int wlength);    ///< ctor

    /**
     * API 결과 구조체의 내용을 채운다.
     * @param  wraw  유니코드 원문
     * @param  wbegins  각 음절별 시작 byte 위치
     * @param  wends  각 음절별 끝 byte 위치
     */
    void organize(const char32_t* wraw, const std::vector<int>& wbegins,
                  const std::vector<int>& wends);

    /**
     * pos_tag_t 타입의 숫자 태그에 대응하는 문자열 태그를 리턴한다.
     * @param  num  숫자 품사 태그
     * @return  문자열 품사 태그
     */
    static const char* pos_str(pos_tag_t num);

    /**
     * 개체명 태그 스트링의 포인터를 전달해서, API 구조체 내 변수에 설정합니다.
     * @param  tag  개체명 태그
     * @return void
     */
    void set_ne_str(const char* tag);

    std::string str();    ///< UTF-8 문자열로 표현합니다.
    std::u32string wstr();    ///< 유니코드 문자열로 표현합니다. (거의) 디버그용

 private:
    std::string _lex;    ///< cache of UTF-8 lexical
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_MORPH_HPP_
