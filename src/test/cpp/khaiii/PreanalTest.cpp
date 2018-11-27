/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */



//////////////
// includes //
//////////////
#include <memory>
#include <string>

#include "cxxopts.hpp"
#include "gtest/gtest.h"

#include "khaiii/Preanal.hpp"
#include "khaiii/Word.hpp"


///////////////
// variables //
///////////////
extern cxxopts::ParseResult *prog_args;    // arguments passed to main program


namespace khaiii {


using std::make_shared;
using std::shared_ptr;
using std::string;
using std::wstring;


//////////////////
// test fixture //
//////////////////
class PreanalTest: public testing::Test {
 public:
    virtual void SetUp() {
        std::string rsc_dir = (*prog_args)["rsc-dir"].as<string>();
        ASSERT_NO_THROW(_preanal.open(rsc_dir));
    }

    virtual void TearDown() {
        ASSERT_NO_THROW(_preanal.close());
    }

 protected:
    Preanal _preanal;

    inline shared_ptr<Word> _apply(wstring raw) {
        auto word = make_shared<Word>(raw.c_str(), raw.length());
        _preanal.apply(word);
        return word;
    }
};


////////////////
// test cases //
////////////////
TEST_F(PreanalTest, apply_exact) {
    // 어절 완전일치 엔트리 "이더리움"에 대해

    auto word1 = _apply(L"이더리움");    // 매칭
    EXPECT_LT(0, word1->char_tags[0]);
    EXPECT_LT(0, word1->char_tags[1]);
    EXPECT_LT(0, word1->char_tags[2]);
    EXPECT_LT(0, word1->char_tags[3]);

    auto word2 = _apply(L"이더리움을");    // 매칭 안됨
    EXPECT_EQ(0, word2->char_tags[0]);
    EXPECT_EQ(0, word2->char_tags[1]);
    EXPECT_EQ(0, word2->char_tags[2]);
    EXPECT_EQ(0, word2->char_tags[3]);
    EXPECT_EQ(0, word2->char_tags[4]);

    auto word3 = _apply(L"이더륨");    // 매칭 안됨
    EXPECT_EQ(0, word3->char_tags[0]);
    EXPECT_EQ(0, word3->char_tags[1]);
    EXPECT_EQ(0, word3->char_tags[2]);

    EXPECT_NO_THROW(_apply(L""));
}


TEST_F(PreanalTest, apply_prefix) {
    // 전망매칭 패턴 "가즈아*"에 대해

    auto word1 = _apply(L"가즈아~");    // 매칭
    EXPECT_LT(0, word1->char_tags[0]);
    EXPECT_LT(0, word1->char_tags[1]);
    EXPECT_LT(0, word1->char_tags[2]);
    EXPECT_EQ(0, word1->char_tags[3]);

    auto word2 = _apply(L"가즈아");    // 매칭
    EXPECT_LT(0, word2->char_tags[0]);
    EXPECT_LT(0, word2->char_tags[1]);
    EXPECT_LT(0, word2->char_tags[2]);

    auto word3 = _apply(L"가자");    // 매칭 안됨
    EXPECT_EQ(0, word3->char_tags[0]);
    EXPECT_EQ(0, word3->char_tags[1]);
}


}    // namespace khaiii
