/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Restore.hpp"


//////////////
// includes //
//////////////
#include <exception>
#include <memory>
#include <vector>
#include <cassert>

#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/Morph.hpp"
#include "khaiii/util.hpp"

/** Supports spdlog::stderr_color_mt */
#include <spdlog/sinks/stdout_color_sinks.h>

namespace khaiii {


using std::exception;
using std::shared_ptr;
using std::string;
using std::vector;


///////////////
// constants //
///////////////
static const char* _B_STRS[POS_TAG_SIZE] = {
     "B-EC",  "B-EF",  "B-EP", "B-ETM", "B-ETN",  "B-IC",  "B-JC", "B-JKB", "B-JKC", "B-JKG",
    "B-JKO", "B-JKQ", "B-JKS", "B-JKV",  "B-JX", "B-MAG", "B-MAJ",  "B-MM", "B-NNB", "B-NNG",
    "B-NNP",  "B-NP",  "B-NR",  "B-SE",  "B-SF",  "B-SH",  "B-SU",  "B-SN",  "B-SO",  "B-SP",
     "B-SS",  "B-SW", "B-SWK",  "B-VA", "B-VCN", "B-VCP",  "B-VV",  "B-VX", "B-XPN",  "B-XR",
    "B-XSA", "B-XSN", "B-XSV",  "B-ZN",  "B-ZV",  "B-ZZ",
};

static const char* _I_STRS[POS_TAG_SIZE] = {
     "I-EC",  "I-EF",  "I-EP", "I-ETM", "I-ETN",  "I-IC",  "I-JC", "I-JKB", "I-JKC", "I-JKG",
    "I-JKO", "I-JKQ", "I-JKS", "I-JKV",  "I-JX", "I-MAG", "I-MAJ",  "I-MM", "I-NNB", "I-NNG",
    "I-NNP",  "I-NP",  "I-NR",  "I-SE",  "I-SF",  "I-SH",  "I-SU",  "I-SN",  "I-SO",  "I-SP",
     "I-SS",  "I-SW", "I-SWK",  "I-VA", "I-VCN", "I-VCP",  "I-VV",  "I-VX", "I-XPN",  "I-XR",
    "I-XSA", "I-XSN", "I-XSV",  "I-ZN",  "I-ZV",  "I-ZZ",
};


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Restore::_log = spdlog::stderr_color_mt("Restore");


////////////////////
// ctors and dtor //
////////////////////
Restore::~Restore() {
    close();
}


/////////////
// methods //
/////////////
std::string chr_tag_t::str() {
    assert(0 < tag && tag <= POS_TAG_SIZE);
    char32_t wstr[2] = {chr, 0};
    const char** table = _B_STRS;
    if (bi == chr_tag_t::I) table = _I_STRS;
    return wstr_to_utf8(wstr) + "/" + table[tag-1];
}


void Restore::open(const char* dir) {
	assert(dir);

	std::string _dir(dir); _dir += "/restore.key";
	_key_mmf.open(_dir.c_str());
	size_t drsz = _dir.length();

	_dir.replace(drsz - 3, 3, "val");
	_val_mmf.open(_dir.c_str());
	assert(_key_mmf.size() * _MAX_VAL_LEN == _val_mmf.size());


	_dir.replace(drsz - 3, 3, "one");
	_one_mmf.open(_dir.c_str());
#ifndef NDEBUG
    for (int i = 0; i < _one_mmf.size(); ++i) {
        SPDLOG_TRACE(_log, "{}: {}, ", i, _one_mmf.data()[i]);
    }
#endif
    _log->info("restore dictionary opened");
}


void Restore::close() {
    _key_mmf.close();
    _val_mmf.close();
    _one_mmf.close();
    _log->debug("restore dictionary closed");
}


vector<chr_tag_t> Restore::restore(char32_t chr, uint16_t tag_out, bool use_dic) const {
    assert(tag_out > 0);
    vector<chr_tag_t> restored;
    if (!is_need_restore(tag_out)) {
        // 원형 복원이 필요없는 경우
        restored.emplace_back(chr_tag_t());
        restored.back().chr = chr;
        restored.back().set_tag(tag_out);
        return restored;
    }

    if (!use_dic) {
        // 원형 복원 사전을 사용하지 않고 첫번째 태그로 바로 부여한다.
        restored.emplace_back(chr_tag_t());
        restored.back().chr = chr;
        restored.back().set_tag(get_one(tag_out));
        return restored;
    }

    int idx = find(chr, tag_out);
    if (idx == -1) {
        // 키가 발견되지 않는 경우 태그 조합 중 첫번째 태그로 부여한다.
        uint16_t tag_one = get_one(tag_out);
#ifndef NDEBUG
        char32_t wstr[2] = {chr, 0};
        _log->info("restore key not found: {}/{} => {}", wstr_to_utf8(wstr), tag_out, tag_one);
#endif
        restored.emplace_back(chr_tag_t());
        restored.back().chr = chr;
        restored.back().set_tag(tag_one);
    } else {
        const uint32_t* val = _val_mmf.data() + (idx * _MAX_VAL_LEN);
        for (int i = 0; *val && i < _MAX_VAL_LEN; ++val, ++i) {
            restored.emplace_back(chr_tag_t());
            restored.back().from_val(*val);
        }
    }
    return restored;
}


bool Restore::is_need_restore(uint16_t tag_out) {
    return tag_out > 2 * POS_TAG_SIZE;
}


int Restore::find(char32_t chr, uint16_t tag_out) const {
    assert(is_need_restore(tag_out));
    uint32_t key = chr << 12 | tag_out;    // key의 경우 12비트를 shift하고 output tag를 합친다.
    const uint32_t* found = reinterpret_cast<const uint32_t*>(
            bsearch(&key, _key_mmf.data(), _key_mmf.size(), sizeof(uint32_t), Restore::key_cmp));
    if (found == nullptr) return -1;
    return found - _key_mmf.data();
}


uint8_t Restore::get_one(uint16_t tag_out) const {
    assert(is_need_restore(tag_out));
    assert(tag_out < _one_mmf.size());
    return _one_mmf.data()[tag_out];
}


int Restore::key_cmp(const void* left, const void* right) {
    uint32_t left_val = *reinterpret_cast<const uint32_t*>(left);
    uint32_t right_val = *reinterpret_cast<const uint32_t*>(right);
    if (left_val < right_val) {
        return -1;
    } else if (left_val > right_val) {
        return 1;
    } else {
        return 0;
    }
}


}    // namespace khaiii
