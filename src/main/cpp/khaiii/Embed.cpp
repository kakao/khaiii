/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Embed.hpp"


/** Supports spdlog::stderr_color_mt */
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

//////////////
// includes //
//////////////
#include <cstdlib>
#include <string>
#include <cassert>

#include "khaiii/Config.hpp"
#ifndef NDEBUG
#include "khaiii/util.hpp"
#endif


namespace khaiii {


using std::shared_ptr;
using std::string;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Embed::_log = spdlog::stderr_color_mt("Embed");


/////////////
// methods //
/////////////
void Embed::open(const Config& cfg, const char* dir) {
	assert(dir);

    _embed_mmf.open(fmt::format("{}/embed.bin", dir).c_str());
    _keys = reinterpret_cast<const char32_t*>(_embed_mmf.data());
    const float* val_start = reinterpret_cast<const float*>(_keys + cfg.vocab_size);
    for (int i = 0; i < cfg.vocab_size; ++i) {
        const float* embed_start = val_start + i * cfg.embed_dim;
        _vals.emplace_back(embedding_t(const_cast<float*>(embed_start), cfg.embed_dim));
        SPDLOG_TRACE(_log, "[{}] {}", i, _vals[i]);
    }
}


void Embed::close() {
    _embed_mmf.close();
}


const embedding_t& Embed::operator[](char32_t chr) const {
    const char32_t* found = reinterpret_cast<const char32_t*>(
            bsearch(&chr, _keys, _vals.size(), sizeof(char32_t), Embed::_key_cmp));
    int idx = 1;    // unknown character index is 1
    if (found != nullptr) idx = found - _keys;
#ifndef NDEBUG
    char32_t wstr[2] = {chr, 0};
    SPDLOG_TRACE(_log, "'{}'({}) {}", wstr_to_utf8(wstr), idx, _vals.at(idx));
#endif
    return _vals.at(idx);
}


const embedding_t& Embed::left_word_bound() const {
    return _vals.at(2);
}


const embedding_t& Embed::right_word_bound() const {
    return _vals.at(3);
}


const embedding_t& Embed::left_padding() const {
    return _vals.at(0);    // padding index is 0 which is zero vector
}


const embedding_t& Embed::right_padding() const {
    return _vals.at(0);    // padding index is 0 which is zero vector
}


int Embed::_key_cmp(const void* left, const void* right) {
	assert(left && right);

    const char32_t* left_ = reinterpret_cast<const char32_t*>(left);
    const char32_t* right_ = reinterpret_cast<const char32_t*>(right);
    return *left_ - *right_;
}


}    // namespace khaiii
