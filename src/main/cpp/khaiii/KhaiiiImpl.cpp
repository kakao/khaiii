/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/KhaiiiImpl.hpp"


//////////////
// includes //
//////////////
#include <exception>
#include <map>
#include <memory>
#include <sstream>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "khaiii/Sentence.hpp"
#include "khaiii/Tagger.hpp"
#include "khaiii/Word.hpp"
#include "khaiii/util.hpp"


namespace khaiii {


using std::dynamic_pointer_cast;
using std::exception;
using std::make_shared;
using std::map;
using std::ostringstream;
using std::recursive_mutex;
using std::shared_ptr;
using std::string;
using std::unique_lock;
using std::vector;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> KhaiiiImpl::_log = spdlog::stderr_color_mt("KhaiiiImpl");


////////////////////
// ctors and dtor //
////////////////////
Except::Except(string msg, const char* file, const int line, const char* func)
        : _msg(msg), _file(file), _line(line), _func(func) {}


KhaiiiImpl::~KhaiiiImpl() {
    close();
}


/////////////
// methods //
/////////////
const char* Except::what() const noexcept {
    return _msg.c_str();
}


string Except::debug() {
    ostringstream oss;
    if (_func != nullptr) oss << _func;
    if (_file != nullptr) {
        oss <<  "(" << _file;
        if (_line > 0) oss << ":" << _line;
        oss << ")";
    }
    if (oss.str().length() > 0) oss << " ";
    oss << _msg;
    return oss.str();
}


shared_ptr<KhaiiiApi> KhaiiiApi::create() {
    return make_shared<KhaiiiImpl>();
}


void KhaiiiImpl::open(string rsc_dir, string opt_str) {
    unique_lock<recursive_mutex> lock(_mutex);
    if (!_is_opened) close();

    rsc_dir = _check_rsc_dir(rsc_dir);

    // load configurations
    // pos-tagger configuration
    _cfg.read_from_file(rsc_dir + "/config.json");
    _cfg.override_from_str(opt_str.c_str());
    // override options passed by argument
    _cfg.override_from_str(opt_str.c_str());

    // open resources
    _rsc.open(_cfg, rsc_dir);
    _is_opened = true;
}


const khaiii_word_t* KhaiiiImpl::analyze(const char* input, const char* opt_str) {
    if (input == nullptr) throw Except("input is nullptr");
    unique_lock<recursive_mutex> lock(_mutex);
    if (!_is_opened) throw Except("handle is not opened");
    auto sent = make_shared<Sentence>(input);
    auto cfg = _cfg.copy_and_override(opt_str);
    Tagger tgr(*cfg, _rsc, sent);
    tgr.tag();
    if (sent->words.size() == 0) return nullptr;
    sent->organize();
    return _deposit_sent(sent);
}


int KhaiiiImpl::analyze_bfr_errpatch(const char* input, const char* opt_str, int16_t* output) {
    if (input == nullptr) throw Except("input is null");
    if (output == nullptr) throw Except("output is null");
    unique_lock<recursive_mutex> lock(_mutex);
    if (!_is_opened) throw Except("handle is not opened");
    auto sent = make_shared<Sentence>(input);
    auto cfg = _cfg.copy_and_override(opt_str);
    cfg->errpatch = false;
    Tagger tgr(*cfg, _rsc, sent);
    tgr.tag();
    int16_t* ptr = output;
    for (auto word : sent->words) {
        if (ptr == output) {
            *ptr = -2;    // BoS
        } else {
            *ptr = -1;    // word delimiter
        }
        ++ptr;
        for (auto char_tag : word->char_tags) {
            *ptr = char_tag;
            ++ptr;
        }
    }
    *ptr = -2;    // EoS
    ++ptr;

#ifndef NDEBUG
    ostringstream oss;
    for (const int16_t* cursor = output; cursor < ptr; ++cursor) {
        if (oss.str().length() > 0) oss << " ";
        oss << *cursor;
    }
    _log->debug("[{}]", oss.str());
#endif

    return ptr - output;
}


void KhaiiiImpl::free_results(const khaiii_word_t* results) {
    if (results == nullptr) return;
    unique_lock<recursive_mutex> lock(_mutex);
    if (!_is_opened) return;
    _withdraw_sent(results);
}


recursive_mutex& KhaiiiImpl::get_mutex() {
    return _mutex;
}


void KhaiiiImpl::close() {
    unique_lock<recursive_mutex> lock(_mutex);
    _rsc.close();
    _is_opened = false;
}


void KhaiiiImpl::set_err_msg(string msg) {
    _err_msg = msg;
}


const char* KhaiiiImpl::get_err_msg() const {
    return _err_msg.c_str();
}


void KhaiiiImpl::set_log_level(string name, string level) {
    map<string, spdlog::level::level_enum> levels = {
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"err", spdlog::level::err},
        {"critical", spdlog::level::critical},
    };

    auto found = levels.find(level);
    if (found == levels.end()) throw Except(fmt::format("invalid log level: {}", level));

    if (name == "all") {
        spdlog::set_level(found->second);
    } else if (name == "") {
        throw Except(fmt::format("no logger name"));
    } else {
        auto logger = spdlog::get(name);
        if (!logger) throw Except(fmt::format("invalid logger name: {}", name));
        logger->set_level(found->second);
    }
}


void KhaiiiImpl::set_log_levels(string name_level_pairs) {
    for (auto name_level_pair : split(name_level_pairs, ',')) {
        auto name_level = split(name_level_pair, ':');
        if (name_level.size() != 2) {
            throw Except(fmt::format("invalid logger name/level pair: {}", name_level_pair));
        }
        KhaiiiImpl::set_log_level(name_level[0], name_level[1]);
    }
}


const khaiii_word_t* KhaiiiImpl::_deposit_sent(shared_ptr<Sentence> sent) {
    const khaiii_word_t* head_word = sent->words[0].get();
    _result_cloakroom[head_word] = sent;
    return head_word;
}


void KhaiiiImpl::_withdraw_sent(const khaiii_word_t* head_word) {
    _result_cloakroom.erase(head_word);
}

string KhaiiiImpl::_check_rsc_dir(string rsc_dir) {
    if (rsc_dir.length() == 0) rsc_dir = fmt::format("{}/share/khaiii", PREFIX);
    if (!file_exists(rsc_dir)) {
        throw Except(fmt::format("resource directory not found: {}", rsc_dir));
    }
    return rsc_dir;
}


}    // namespace khaiii
