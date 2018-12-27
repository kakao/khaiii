/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Config.hpp"


//////////////
// includes //
//////////////
#include <fstream>

#include "fmt/format.h"
#include "nlohmann/json.hpp"

#include "khaiii/KhaiiiApi.hpp"


namespace khaiii {


using std::exception;
using std::ifstream;
using std::make_shared;
using std::shared_ptr;
using std::string;


/////////////
// methods //
/////////////
void Config::read_from_file(string path) {
    try {
        ifstream ifs(path);
        nlohmann::json jsn;
        ifs >> jsn;
        set_members(jsn);
    } catch (const exception& exc) {
        throw Except(fmt::format("fail to parse config: {}", exc.what()));
    }
}


void Config::override_from_str(const char* opt_str) {
    if (opt_str == nullptr || opt_str[0] == '\0') return;

    try {
        auto jsn = nlohmann::json::parse(opt_str);
        override_members(jsn);
    } catch (const exception& exc) {
        throw Except(fmt::format("fail to parse option: {}\n{}", exc.what(), opt_str));
    }
}


Config* Config::copy_and_override(const char* opt_str) {
    if (opt_str == nullptr || opt_str[0] == '\0') return this;

    auto found = _cfg_cache.find(opt_str);
    if (found != _cfg_cache.end()) return found->second.get();

    auto cfg = copy();
    try {
        auto jsn = nlohmann::json::parse(opt_str);
        cfg->override_members(jsn);
        _cfg_cache[opt_str] = cfg;
    } catch (const exception& exc) {
        throw Except(fmt::format("fail to parse option: {}\n{}", exc.what(), opt_str));
    }

    return cfg.get();
}


void Config::set_members(const nlohmann::json& jsn) {
    class_num = jsn.value("class_num", class_num);
    if (class_num <= 0) throw Except(fmt::format("invalid 'class_num' value: {}", class_num));

    embed_dim = jsn.value("embed_dim", embed_dim);
    if (embed_dim <= 0) throw Except(fmt::format("invalid 'embed_dim' value: {}", embed_dim));

    hidden_dim = jsn.value("hidden_dim", hidden_dim);
    if (hidden_dim <= 0) throw Except(fmt::format("invalid 'hidden_dim' value: {}", hidden_dim));

    vocab_size = jsn.value("vocab_size", vocab_size);
    if (vocab_size <= 0) throw Except(fmt::format("invalid 'vocab_size' value: {}", vocab_size));

    window = jsn.value("window", window);
    if (window <= 0) throw Except(fmt::format("invalid 'window' value: {}", window));

    override_members(jsn);
}

void Config::override_members(const nlohmann::json& jsn) {
    preanal = jsn.value("preanal", preanal);
    errpatch = jsn.value("errpatch", errpatch);
    restore = jsn.value("restore", restore);
}

shared_ptr<Config> Config::copy() {
    auto that = make_shared<Config>();
    that->class_num = class_num;
    that->embed_dim = embed_dim;
    that->hidden_dim = hidden_dim;
    that->vocab_size = vocab_size;
    that->window = window;
    that->preanal = preanal;
    that->errpatch = errpatch;
    that->restore = restore;
    return that;
}


}    // namespace khaiii
