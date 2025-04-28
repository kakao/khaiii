/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */



//////////////
// includes //
//////////////
#include <array>

#include "khaiii/khaiii_dev.h"

#include "khaiii/ErrPatch.hpp"
#include "khaiii/KhaiiiApiTest.hpp"
#include "khaiii/util.hpp"

using std::array;
using std::string;


//////////////////
// test fixture //
//////////////////
class KhaiiiDevTest: public KhaiiiApiTest {};


////////////////
// test cases //
////////////////
TEST_F(KhaiiiDevTest, analyze_bfr_errorpatch) {
    array<int16_t, 13> output;
    EXPECT_EQ(13, khaiii_analyze_bfr_errpatch(_handle, u8"진정한 테스트입니다.", "", &output[0]));
    EXPECT_EQ(khaiii::ErrPatch::SENT_DELIM_NUM, static_cast<wchar_t>(output[0]));    // bos/eos
    EXPECT_EQ(khaiii::ErrPatch::WORD_DELIM_NUM, static_cast<wchar_t>(output[4]));    // bow/eow
    EXPECT_EQ(khaiii::ErrPatch::SENT_DELIM_NUM, static_cast<wchar_t>(output[12]));   // bos/eos

    EXPECT_GT(0, khaiii_analyze_bfr_errpatch(-1, u8"", "", &output[0]));    // invalid handle
    EXPECT_GT(0, khaiii_analyze_bfr_errpatch(_handle, nullptr, "", &output[0]));    // null input
    EXPECT_GT(0, khaiii_analyze_bfr_errpatch(_handle, u8"", "", nullptr));    // null output
}


TEST_F(KhaiiiDevTest, set_log_level) {
    EXPECT_EQ(0, khaiii_set_log_level("all", "trace"));
    EXPECT_EQ(0, khaiii_set_log_level("all", "debug"));
    EXPECT_EQ(0, khaiii_set_log_level("all", "info"));
    EXPECT_EQ(0, khaiii_set_log_level("all", "warn"));
    EXPECT_EQ(0, khaiii_set_log_level("all", "err"));
    EXPECT_EQ(0, khaiii_set_log_level("all", "critical"));

    EXPECT_GT(0, khaiii_set_log_level(nullptr, "debug"));    // null logger
    EXPECT_GT(0, khaiii_set_log_level("", "debug"));    // zero string logger
    EXPECT_GT(0, khaiii_set_log_level("__invalid_logger__", "debug"));
    EXPECT_GT(0, khaiii_set_log_level("Tagger", nullptr));    // null level
    EXPECT_GT(0, khaiii_set_log_level("Tagger", ""));    // zero string level
    EXPECT_GT(0, khaiii_set_log_level("Tagger", "__invalid_level__"));
}


TEST_F(KhaiiiDevTest, set_log_levels) {
    EXPECT_EQ(0, khaiii_set_log_levels("all:warn,Tagger:info"));
    EXPECT_EQ(0, khaiii_set_log_levels(""));    // zero name/level pair

    EXPECT_GT(0, khaiii_set_log_levels(nullptr));    // null name/level pair
    EXPECT_GT(0, khaiii_set_log_levels("all,Tagger:info"));    // invalid format
}
