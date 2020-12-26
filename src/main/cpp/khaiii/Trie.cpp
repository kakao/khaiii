/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Trie.hpp"


//////////////
// includes //
//////////////
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <list>
#include <memory>
#include <string>

#include "boost/lexical_cast.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/util.hpp"


namespace khaiii {


using std::exception;
using std::find_if;
using std::list;
using std::shared_ptr;
using std::string;
using std::wstring;

using boost::optional;


///////////////
// constants //
///////////////
static const size_t _LIN_BIN_NUM = 32;    // linear/binary search 경계가 되는 element 갯수


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Trie::_log = spdlog::stderr_color_mt("Trie");


////////////////////
// ctors and dtor //
////////////////////
Trie::~Trie() {
    close();
}


/////////////
// methods //
/////////////
void Trie::open(string path) {
    _mmf.open(path);

#ifndef NDEBUG
    const _node_t* root_node = _mmf.data();
    for (int i = 0; i < sizeof(root_node) / sizeof(_node_t); ++i) {
        SPDLOG_TRACE(_log, root_node[i].str(root_node));
    }
#endif
}


void Trie::close() {
    _mmf.close();
}


/*
optional<uint32_t> Trie::find(const wstring& key) const {
    return find(key.c_str());
}


optional<uint32_t> Trie::find(const wchar_t* key) const {
    assert(key != nullptr);
    if (*key == L'\0') return boost::none;
    return _find(key, _mmf.data());
}
*/


list<Trie::match_t> Trie::search_common_prefix_matches(const wstring& text, int max_len) const {
    return search_common_prefix_matches(text.c_str(), max_len);
}


list<Trie::match_t> Trie::search_common_prefix_matches(const wchar_t* text, int max_len) const {
    assert(text != nullptr);
    list<match_t> found;
    _search(text, _mmf.data(), &found, 0, max_len);
    return found;
}


optional<Trie::match_t> Trie::search_longest_prefix_match(const wchar_t* text, int max_len) const {
    list<match_t> found = search_common_prefix_matches(text, max_len);
    if (found.empty()) return boost::none;
    return optional<match_t>(found.back());
}


/*
boost::optional<uint32_t> Trie::_find(const wchar_t* key, const _node_t* node) const {
    SPDLOG_TRACE(_log, "key: [{}], {}", key, node->str(_data()));
    if (node->child_start <= 0 || node->child_num <= 0) return boost::none;
    auto begin = node + node->child_start;
    auto end = begin + node->child_num;
    auto found_node = end;
    if (node->child_num < _LIN_BIN_NUM) {
        // linear search
        auto pred = [&key] (const _node_t& _node) { return _node.chr == *key; };
        found_node = find_if(begin, end, pred);
    } else {
        // binary search
        _node_t key_node;
        key_node.chr = *key;
        void* found_ptr = ::bsearch(&key_node, begin, end - begin, sizeof(_node_t), _node_t::cmp);
        if (found_ptr) found_node = static_cast<const _node_t*>(found_ptr);
    }
    if (found_node == end) {
        SPDLOG_TRACE(_log, "  not found");
        return boost::none;
    } else {
        SPDLOG_TRACE(_log, "  found: {}", found_node->str(_data()));
        key += 1;
        if (*key == L'\0') {
            if (found_node->val > 0) {
                return optional<uint32_t>(found_node->val);
            } else {
                return boost::none;
            }
        } else {
            return _find(key, found_node);
        }
    }
}
*/


void Trie::_search(const wchar_t* text, const _node_t* node, list<Trie::match_t>* matches,
                   int len, int max_len) const {
    SPDLOG_TRACE(_log, "text({}): [{}], {}", len, wstr_to_utf8(text), node->str(_data()));
    if (*text == '\0' || len > max_len || node->child_start <= 0 || node->child_num <= 0) return;
    auto begin = node + node->child_start;
    auto end = begin + node->child_num;
    auto found_node = end;
    if (node->child_num < _LIN_BIN_NUM) {
        // linear search
        auto pred = [&text] (const _node_t& _node) { return _node.chr == *text; };
        found_node = find_if(begin, end, pred);
    } else {
        // binary search
        _node_t key_node;
        key_node.chr = *text;
        void* found_ptr = ::bsearch(&key_node, begin, end - begin, sizeof(_node_t), _node_t::cmp);
        if (found_ptr) found_node = static_cast<const _node_t*>(found_ptr);
    }
    if (found_node == end) {
        SPDLOG_TRACE(_log, "  not matched");
        return;
    } else {
        SPDLOG_TRACE(_log, "  matched: {}", found_node->str(_data()));
        if (found_node->val > 0) matches->emplace_back(len + 1, found_node->val);
        _search(text + 1, found_node, matches, len + 1, max_len);
    }
}


}    // namespace khaiii
