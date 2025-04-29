/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Sentence.hpp"


//////////////
// includes //
//////////////
#include <cassert>
#include <cctype>
#include <iomanip>
#include <locale>
#include <sstream>
#include <charconv>
#include <cuchar>

#include "khaiii/Word.hpp"
#include "khaiii/util.hpp"


/** Supports spdlog::stderr_color_mt */
#include <spdlog/sinks/stdout_color_sinks.h>

namespace khaiii {


using std::codecvt;
using std::codecvt_base;
using std::dec;
using std::hex;
using std::locale;
using std::make_shared;
using std::mbstate_t;
using std::setfill;
using std::setw;
using std::shared_ptr;
using std::string;
using std::use_facet;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Sentence::_log = spdlog::stderr_color_mt("Sentence");


////////////////////
// ctors and dtor //
////////////////////
Sentence::Sentence(const char* raw): _raw(raw), _morph_cnt(0) {
    _characterize();
    _tokenize();
}


/////////////
// methods //
/////////////
void Sentence::organize() {
    for (int i = 0; i < words.size(); ++i) {
        if (i > 0) words[i-1]->next = words[i].get();
        words[i]->organize(_wraw.c_str(), _wbegins, _wends);
#ifndef NDEBUG
        _log->debug("[{}] word: {}", i, words[i]->str());
        for (int j = 0; j < words[i]->morph_vec.size(); ++j) {
            _log->debug("\t[{}] morph: {}", j, words[i]->morph_vec[j]->str());
        }
#endif
    _morph_cnt += words[i]->morph_vec.size();
    }
}


int Sentence::get_lwb_delta(int wrd_idx, int chr_idx) {
    assert(0 <= chr_idx && chr_idx < words[wrd_idx]->wlength);
    return -chr_idx;
}


int Sentence::get_rwb_delta(int wrd_idx, int chr_idx) {
    assert(0 <= chr_idx && chr_idx < words[wrd_idx]->wlength);
    return words[wrd_idx]->wlength - chr_idx - 1;
}


void Sentence::_tokenize() {
    bool is_in_space = true;
    for (int idx = 0; idx < _wraw.size(); ++idx) {
        if (is_space(_wraw[idx])) {
            is_in_space = true;
        } else {
            if (is_in_space) {
                // first character => start of word
                words.emplace_back(make_shared<Word>(&_wraw[idx], 1));
            } else {
                words.back()->wlength += 1;
            }
            is_in_space = false;
        }
    }

    for (auto& word : words) {
        word->set_begin_length(_wraw.c_str(), _wbegins, _wends);
        _log->debug("'{}'{}~{}|{},{}", word->str(), word->begin, word->length,
                    (word->wbegin - &_wraw[0]), word->wlength);
    }
}


#if defined(_WIN32) and 0
/** It will set locale temporalily to locale.C. */
static struct sLocale {
    std::locale previous;
    static constexpr const char* tar = "";
    inline sLocale() {
	    SetConsoleOutputCP(CP_UTF8);
	    SetConsoleCP(CP_UTF8);
    }
} __sLocale;
#endif

void Sentence::_characterize() {
    assert(_raw != nullptr);
    const std::codecvt<char32_t, char, mbstate_t>* facet = &use_facet<codecvt<char32_t, char, mbstate_t>>(
        std::locale::classic()
    );
    auto mbst = mbstate_t();
    const char* from_next = nullptr;
    std::u32stringstream wss;
    for (const char* from_curr = _raw; *from_curr != '\0'; from_curr = from_next) {
        char32_t wchar[2] = U"";
        char32_t* to_next = nullptr;
        auto result = facet->in(mbst, from_curr, from_curr + 6, from_next, wchar, wchar + 1,
                               to_next);
        assert(result == codecvt_base::partial || result == codecvt_base::ok);
        wss << wchar[0];
        _wbegins.emplace_back(from_curr - _raw);
        _wends.emplace_back(from_next - _raw);
        // _log->debug("'{}'({}){}~{}|{}~{}", string(from_curr, from_next - from_curr), hex,
        //             static_cast<int>(wchar[0]), dec, (from_curr - _raw), (from_next - _raw));
    }
    _wraw = wss.str();
    assert(_wraw.length() == _wbegins.size());
    assert(_wraw.length() == _wends.size());
}


}    // namespace khaiii
