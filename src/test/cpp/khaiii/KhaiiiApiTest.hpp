/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_TEST_CPP_KHAIII_KHAIIIAPITEST_HPP_
#define SRC_TEST_CPP_KHAIII_KHAIIIAPITEST_HPP_



//////////////
// includes //
//////////////
#include <string>

#include "gtest/gtest.h"

#include "khaiii/khaiii_api.h"


//////////////////
// test fixture //
//////////////////
class KhaiiiApiTest: public testing::Test {
 public:
    virtual void SetUp();    ///< set up
    virtual void TearDown();    ///< tear down

 protected:
    int _handle = -1;    ///< 핸들

    /**
     * 어절의 분석 결과를 비교하기위한 함수 (포지션 정보 포함)
     * @param  expected  기대하는 결과 문자열. 예: "[1:7]\t안녕/IC[1:6] + ?/SF[7:1]"
     * @param  actual  실제 어절 결과
     */
    void _expect_eq_word(std::string expected, const khaiii_word_t& actual) const;

    /**
     * 어절의 분석 결과 중 형태소 부분만을 비교하기 위한 함수
     * @param  expected  기대하는 결과 문자열. 예: "안녕/IC + ?/SF"
     * @param  actual  실제 어절 결과
     */
    void _expect_eq_morphs(std::string expected, const khaiii_word_t& actual) const;
};


#endif    // SRC_TEST_CPP_KHAIII_KHAIIIAPITEST_HPP_
