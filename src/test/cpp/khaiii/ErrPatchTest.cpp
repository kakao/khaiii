/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */



//////////////
// includes //
//////////////
#include <memory>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

#include "khaiii/ErrPatch.hpp"
#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/Word.hpp"
#include "khaiii/util.hpp"


///////////////
// variables //
///////////////
extern cxxopts::ParseResult* prog_args;    // arguments passed to main program


namespace khaiii {


using std::make_shared;
using std::ostringstream;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;
using std::wstring;


//////////////////
// test fixture //
//////////////////
class ErrPatchTest: public testing::Test {
 public:
    virtual void SetUp() {
        std::string rsc_dir = (*prog_args)["rsc-dir"].as<string>();
        ASSERT_NO_THROW(_khaiii_api->open(rsc_dir, "{\"errpatch\": false}"));
    }

    virtual void TearDown() {
        ASSERT_NO_THROW(_khaiii_api->close());
    }

 protected:
    static shared_ptr<spdlog::logger> _log;    ///< logger

    shared_ptr<KhaiiiApi> _khaiii_api = KhaiiiApi::create();

    void _check(string raw, string left, string right) {
        auto bfr = _khaiii_api->analyze(raw.c_str(), "{\"errpatch\": false}");
        string bfr_str = _to_str(bfr);
        if (left != bfr_str) {
            _log->warn("error not found: '{}' => E:'{}' vs A:'{}'", raw, left, bfr_str);
            return;
        }
        auto aft = _khaiii_api->analyze(raw.c_str(), "{\"errpatch\": true}");
        EXPECT_STREQ(right.c_str(), _to_str(aft).c_str());
    }

    string _to_str(const khaiii_word_t* results) {
        ostringstream oss;
        for (auto word = results; word != nullptr; word = word->next) {
            if (word != results) oss << " + _ + ";
            const khaiii_morph_t* morphs = word->morphs;
            for (auto morph = morphs; morph != nullptr; morph = morph->next) {
                if (morph != morphs) oss << " + ";
                oss << morph->lex << "/" << morph->tag;
            }
        }
        return oss.str();
    }
};


shared_ptr<spdlog::logger> ErrPatchTest::_log = spdlog::stderr_color_mt("ErrPatchTest");


////////////////
// test cases //
////////////////
TEST_F(ErrPatchTest, apply) {
    // for base model
    _check("지저스크라이스트", "지저스크라이스/NNP + 트/NNG", "지저스/NNP + 크라이스트/NNP");
    _check("지저스 크라이스트", "지저스/NNP + _ + 크라이스/NNP + 트/NNG",
           "지저스/NNP + _ + 크라이스트/NNP");
    _check("고타마싯다르타", "고타마싯다르타/NNP", "고타마/NNP + 싯다르타/NNP");
    _check("무함마드압둘라", "무함마드압/NNP + 둘/NR + 라/NNP", "무함마드/NNP + 압둘라/NNP");

    /*
    // for large model
    _check("지저스크라이스트", "지/NNG + 저스크라이스/NNP + 트/NNG", "지저스/NNP + 크라이스트/NNP");
    _check("지저스 크라이스트", "지저스/NNP + _ + 크라이스/NNP + 트/NNG",
           "지저스/NNP + _ + 크라이스트/NNP");
    _check("고타마싯다르타", "고타마싯다르타/NNP", "고타마/NNP + 싯다르타/NNP");
    _check("무함마드압둘라", "무함마드압둘라/NNP", "무함마드/NNP + 압둘라/NNP");
    */
}


}    // namespace khaiii
