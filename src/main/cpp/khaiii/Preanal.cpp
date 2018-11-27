/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Preanal.hpp"


//////////////
// includes //
//////////////
#include <exception>

#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/Word.hpp"


namespace khaiii {


using std::exception;
using std::shared_ptr;
using std::string;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Preanal::_log = spdlog::stderr_color_mt("Preanal");


////////////////////
// ctors and dtor //
////////////////////
Preanal::~Preanal() {
    close();
}


/////////////
// methods //
/////////////
void Preanal::open(string dir) {
    _trie.open(dir + "/preanal.tri");
    _val_mmf.open(dir + "/preanal.val");
    _log->info("preanal dictionary opened");
}


void Preanal::close() {
    _trie.close();
    _val_mmf.close();
    _log->debug("preanal dictionary closed");
}


void Preanal::apply(shared_ptr<Word> word) const {
    auto matches = _trie.search_common_prefix_matches(word->wbegin, word->wlength);
    int len = 0;
    int idx = -1;
    for (auto match = matches.rbegin(); match != matches.rend(); ++match) {
        bool is_exact = match->val % 2 == 0;
        if (is_exact && match->len == word->wlength) {
            len = match->len;
            idx = match->val / 2;
        } else if (!is_exact) {
            len = match->len;
            idx = (match->val - 1) / 2;
        }
        if (len > 1 && idx >= 0) break;
    }
    if (len <= 0 || idx < 0) return;
    const uint16_t* tag_out_start = &_val_mmf.data()[idx];
    for (int i = 0; i < len; ++i) {
        word->char_tags[i] = tag_out_start[i];
    }
}


}    // namespace khaiii
