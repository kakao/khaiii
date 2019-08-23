/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/KhaiiiApiTest.hpp"


//////////////
// includes //
//////////////
#include <sstream>

#include "cxxopts.hpp"

#include "khaiii/KhaiiiApi.hpp"


///////////////
// variables //
///////////////
extern cxxopts::ParseResult* prog_args;    // arguments passed to main program


using std::string;
using std::ostringstream;


/////////////
// methods //
/////////////
void KhaiiiApiTest::SetUp() {
    string rsc_dir = (*prog_args)["rsc-dir"].as<string>();
    _handle = khaiii_open(rsc_dir.c_str(), "");
    ASSERT_GT(_handle, 0);
}

void KhaiiiApiTest::TearDown() {
    ASSERT_NO_THROW(khaiii_close(_handle));
}

void KhaiiiApiTest::_expect_eq_word(string expected, const khaiii_word_t& actual) const {
    ostringstream oss;
    oss << "[" << actual.begin << ":" << actual.length << "]\t";
    for (auto morph = actual.morphs; morph; morph = morph->next) {
        if (morph != actual.morphs) oss << " + ";
        oss << morph->lex << "/" << morph->tag
            << "[" << morph->begin << ":" << morph->length << "]";
    }
    EXPECT_STREQ(expected.c_str(), oss.str().c_str());
}

/**
 * 어절의 분석 결과 중 형태소 부분만을 비교하기 위한 함수
 * @param  expected  기대하는 결과 문자열. 예: "안녕/IC + ?/SF"
 * @param  actual  실제 어절 결과
 */
void KhaiiiApiTest::_expect_eq_morphs(string expected, const khaiii_word_t& actual) const {
    ostringstream oss;
    for (auto morph = actual.morphs; morph; morph = morph->next) {
        if (morph != actual.morphs) oss << " + ";
        oss << morph->lex << "/" << morph->tag;
    }
    EXPECT_STREQ(expected.c_str(), oss.str().c_str());
}


////////////////
// test cases //
////////////////
TEST_F(KhaiiiApiTest, version) {
    EXPECT_STREQ(KHAIII_VERSION, khaiii_version());
}


TEST_F(KhaiiiApiTest, open_close) {
    string rsc_dir = (*prog_args)["rsc-dir"].as<string>();

    int handle = khaiii_open(rsc_dir.c_str(), "");
    EXPECT_LT(0, handle) << "rsc_dir: " << rsc_dir;
    EXPECT_NO_THROW(khaiii_close(handle));

    // file, not dir
    EXPECT_GT(0, khaiii_open((rsc_dir + "/config.json").c_str(), ""));
    EXPECT_GT(0, khaiii_open((rsc_dir + "/..").c_str(), ""));    // invalid directory
    EXPECT_GT(0, khaiii_open((rsc_dir + "/__not_existing_dir__").c_str(), ""));
    EXPECT_GT(0, khaiii_open(nullptr, ""));

    khaiii_close(-1);    // invalid handle
    string err_msg = khaiii_last_error(-1);
    EXPECT_LT(0, err_msg.length());    // must be some error message
}


TEST_F(KhaiiiApiTest, analyze) {
    auto results = khaiii_analyze(_handle, u8"\v안녕! \t새로운 세상~\n", "");

    auto word1 = results;
    EXPECT_NE(nullptr, word1);
    _expect_eq_word(u8"[1:7]\t안녕/IC[1:6] + !/SF[7:1]", *word1);

    auto word2 = word1->next;
    EXPECT_NE(nullptr, word2);
    _expect_eq_word(u8"[10:9]\t새롭/VA[10:6] + ㄴ/ETM[16:3]", *word2);

    auto word3 = word2->next;
    EXPECT_NE(nullptr, word3);
    _expect_eq_word(u8"[20:7]\t세상/NNG[20:6] + ~/SO[26:1]", *word3);

    khaiii_free_results(_handle, results);

    results = khaiii_analyze(-1, "", "");    // invalid handle
    string err_msg = khaiii_last_error(-1);
    EXPECT_LT(0, err_msg.length());    // must be some error message

    results = khaiii_analyze(_handle, nullptr, "");    // null input
    err_msg = khaiii_last_error(_handle);
    EXPECT_LT(0, err_msg.length());    // must be some error message
}


TEST_F(KhaiiiApiTest, free_results) {
    auto results = khaiii_analyze(_handle, u8"\v안녕? \t새로운 세상~\n", "");
    EXPECT_NO_THROW(khaiii_free_results(_handle, results));
    EXPECT_NO_THROW(khaiii_free_results(_handle, nullptr));

    khaiii_free_results(-1, nullptr);    // invalid handle
    string err_msg = khaiii_last_error(-1);
    EXPECT_LT(0, err_msg.length());    // must be some error message
}


TEST_F(KhaiiiApiTest, last_error) {
    EXPECT_GT(0, khaiii_open("/__not_existing_dir__", ""));
    string err_msg = khaiii_last_error(-1);
    EXPECT_LT(0, err_msg.length());
}


TEST_F(KhaiiiApiTest, restore_true) {
    auto result1 = khaiii_analyze(_handle, u8"됐어", "");
    EXPECT_NE(nullptr, result1);
    _expect_eq_morphs(u8"되/VV + 었/EP + 어/EC", *result1);
    khaiii_free_results(_handle, result1);

    auto result2 = khaiii_analyze(_handle, u8"사랑해", "");
    EXPECT_NE(nullptr, result2);
    _expect_eq_morphs(u8"사랑/NNG + 하/XSV + 여/EC", *result2);
    khaiii_free_results(_handle, result2);

    auto result3 = khaiii_analyze(_handle, u8"먹혀", "");
    EXPECT_NE(nullptr, result3);
    _expect_eq_morphs(u8"먹히/VV + 어/EC", *result3);
    khaiii_free_results(_handle, result3);

    auto result4 = khaiii_analyze(_handle, u8"보여줄", "");
    EXPECT_NE(nullptr, result4);
    _expect_eq_morphs(u8"보이/VV + 어/EC + 주/VX + ㄹ/ETM", *result4);
    khaiii_free_results(_handle, result4);
}


TEST_F(KhaiiiApiTest, restore_false) {
    auto result1 = khaiii_analyze(_handle, u8"됐어", "{\"restore\": false}");
    EXPECT_NE(nullptr, result1);
    _expect_eq_morphs(u8"됐/VV + 어/EC", *result1);
    khaiii_free_results(_handle, result1);

    auto result2 = khaiii_analyze(_handle, u8"사랑해", "{\"restore\": false}");
    EXPECT_NE(nullptr, result2);
    _expect_eq_morphs(u8"사랑/NNG + 해/XSV", *result2);
    khaiii_free_results(_handle, result2);

    auto result3 = khaiii_analyze(_handle, u8"먹혀", "{\"restore\": false}");
    EXPECT_NE(nullptr, result3);
    _expect_eq_morphs(u8"먹혀/VV", *result3);
    khaiii_free_results(_handle, result3);

    auto result4 = khaiii_analyze(_handle, u8"보여줄", "{\"restore\": false}");
    EXPECT_NE(nullptr, result4);
    _expect_eq_morphs(u8"보여/VV + 줄/VX", *result4);
    khaiii_free_results(_handle, result4);
}
