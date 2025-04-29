/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Word.hpp"


//////////////
// includes //
//////////////
#include <string>
#include <vector>
#include <cassert>

#include "khaiii/Morph.hpp"
#include "khaiii/util.hpp"


namespace khaiii {


using std::make_shared;
using std::string;
using std::vector;
using std::u32string;


////////////////////
// ctors and dtor //
////////////////////
Word::Word(const char32_t* wbegin, int wlength): wbegin(wbegin), wlength(wlength),
                                                char_tags(wlength) {
    begin = -1;
    length = -1;
    morphs = nullptr;
    next = nullptr;
}


/////////////
// methods //
/////////////
void Word::set_begin_length(const char32_t* wchars, const vector<int> &wbegins,
                            const vector<int> &wends) {
	assert(wchars);

    int wbegin_idx = wbegin - wchars;
    begin = wbegins.at(wbegin_idx);
    length = wends.at(wbegin_idx + wlength - 1) - begin;
    char_tags.resize(wlength);
}


void Word::set_embeds(const Resource& rsc) {
    embeds.reserve(wlength);
    for (int i = 0; i < wlength; ++i) embeds.emplace_back(rsc.embed[*(wbegin + i)]);
}


void Word::add_morph(const std::u32stringstream& wlex, uint8_t tag, int begin_idx, int end_idx) {
    const char32_t* morph_wbegin = wbegin + begin_idx;
    int morph_wlength = end_idx - begin_idx + 1;
    morph_vec.emplace_back(
		    make_shared<Morph>(wlex.str().c_str(), static_cast<pos_tag_t>(tag), morph_wbegin,
                           morph_wlength));
}


void Word::organize(const char32_t* wraw, const vector<int>& wbegins, const vector<int>& wends) {
    for (int i = 0; i < morph_vec.size(); ++i) {
        if (i > 0) morph_vec[i-1]->next = morph_vec[i].get();
        morph_vec[i]->organize(wraw, wbegins, wends);
    }
}


void Word::make_morphs() {
    std::u32stringstream wlex;
    uint8_t tag = 0;
    int begin_idx = -1;
    int end_idx = -1;
    for (int i = 0; i < restored.size(); ++i) {
        for (auto chr : restored[i]) {
            if (chr.bi == chr_tag_t::I && chr.tag == tag) {
                // 이전 형태소의 연속이므로 새로 생성하지 않고 추가해준다.
                wlex << chr.chr;
                end_idx = i;
            } else {
                if (wlex.str().length() > 0) add_morph(wlex, tag, begin_idx, end_idx);
                wlex.str(U"");
                wlex << chr.chr;
                tag = chr.tag;
                begin_idx = i;
                end_idx = i;
            }
        }
    }
    if (wlex.str().length() > 0) add_morph(wlex, tag, begin_idx, end_idx);

    // linked-list 포인터들을 연결해준다.
    morphs = morph_vec[0].get();
    for (int i = 0; i < morph_vec.size() - 1; ++i) {
        morph_vec[i]->next = morph_vec[i+1].get();
    }
}


string Word::str() const {
    return wstr_to_utf8(wstr().c_str());
}


u32string Word::wstr() const {
    std::u32stringstream wss;
    wss << u32string(wbegin, wlength) << U":" << begin << U"," << length;
    return wss.str();
}


}    // namespace khaiii
