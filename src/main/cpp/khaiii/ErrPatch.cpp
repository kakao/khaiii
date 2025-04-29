/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */

/** Supports spdlog::stderr_color_mt */
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "khaiii/ErrPatch.hpp"


//////////////
// includes //
//////////////
#include <exception>
#include <memory>
#include <vector>
#include <cassert>

#include "khaiii/Sentence.hpp"
#include "khaiii/Word.hpp"


namespace khaiii {


using std::dynamic_pointer_cast;
using std::exception;
using std::shared_ptr;
using std::string;
using std::vector;


////////////////////
// static members //
////////////////////
const char32_t ErrPatch::WORD_DELIM_NUM = -1;
const char32_t ErrPatch::SENT_DELIM_NUM = -2;

shared_ptr<spdlog::logger> ErrPatch::_log = spdlog::stderr_color_mt("ErrPatch");


////////////////////
// ctors and dtor //
////////////////////
ErrPatch::~ErrPatch() {
    close();
}


/////////////
// methods //
/////////////
void ErrPatch::open(const char* dir) {
	assert(dir);

	/** @brief making new string is needed. */
	std::string __dir(dir);
	__dir += "/errpatch.tri";
	
	size_t drsz = __dir.length();
	_trie.open(__dir.c_str());

	__dir.replace(drsz - 3, 3, "val");
	_val_mmf.open(__dir.c_str());
	MemMapFile<uint8_t> len_mmf;

	__dir.replace(drsz - 3, 3, "len");
	/** 각 value들의 길이 정보 */
	len_mmf.open(__dir.c_str());

	_vals.reserve(len_mmf.size());
	const uint8_t* lens = len_mmf.data();
	const int16_t* val_ptr = _val_mmf.data();

	for (int i = 0; i < len_mmf.size(); ++i) {
		/* 길이 정보를 이용하여 int16_t 가변길이 배열인 값(_vals)을 세팅한다. */
		_vals.emplace_back(val_ptr);
		val_ptr += lens[i] + 1;    /* 길이 + 마지막 0 */
	}

	assert(_vals.size() == len_mmf.size());
	assert(val_ptr - _val_mmf.data() == _val_mmf.size());

	_log->info("errpatch dictionary opened");
}


void ErrPatch::close() {
    _trie.close();
    _val_mmf.close();
    _log->debug("errpatch dictionary closed");
}


void ErrPatch::apply(shared_ptr<Sentence> sent) const {
    vector<uint16_t*> outputs;    // 매칭된 패치의 정분석 결과 태그 값을 덮어쓸 출력 위치
    vector<char32_t> chars = _get_char_tag_mixes(sent, &outputs);

    for (int i = 0; i < chars.size(); ++i) {
        auto found = _trie.search_longest_prefix_match(&chars[i]);
        if (found == boost::none) continue;
        auto val = _vals[found->val];
        for (int j = 0; j < found->len; ++j) {
            if (outputs[i + j] == nullptr) {
                assert(val[j] == WORD_DELIM_NUM || val[j] == SENT_DELIM_NUM);
                continue;
            }
            *outputs[i + j] = val[j];
        }
        i += found->len - 1;
    }
}


vector<char32_t> ErrPatch::_get_char_tag_mixes(shared_ptr<Sentence> sent,
                                              vector<uint16_t*>* outputs) {
    vector<char32_t> chars;
    chars.reserve(2 + 2 * sent->words.size());
    outputs->reserve(2 + 2 * sent->words.size());
    chars.emplace_back(SENT_DELIM_NUM);    // 문장 경계
    outputs->emplace_back(nullptr);
    for (auto& word : sent->words) {
        if (chars.size() > 1) {
            chars.emplace_back(WORD_DELIM_NUM);    // 어절 경계
            outputs->emplace_back(nullptr);
        }
        for (int i = 0; i < word->wlength; ++i) {
            char32_t char_tag_mix = (word->wbegin[i] << 12) | word->char_tags[i];
            _log->debug("{:5x}|{:3x} -> {:08x}", static_cast<int>(word->wbegin[i]),
                        word->char_tags[i], static_cast<int>(char_tag_mix));
            chars.emplace_back(char_tag_mix);
            outputs->emplace_back(&word->char_tags[i]);
        }
    }
    chars.emplace_back(SENT_DELIM_NUM);    // 문장 경계
    outputs->emplace_back(nullptr);
    chars.emplace_back(0);    // 마지막 string termination
    outputs->emplace_back(nullptr);
    return chars;
}


}    // namespace khaiii
