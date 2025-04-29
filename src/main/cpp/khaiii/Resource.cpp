/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Resource.hpp"

/** Supports spdlog::stderr_color_mt */
#include <spdlog/sinks/stdout_color_sinks.h>

//////////////
// includes //
//////////////
#include <exception>
#include <memory>
#include <cassert>

#include "khaiii/Config.hpp"
#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/nn/tensor.hpp"


namespace khaiii {


using std::exception;
using std::shared_ptr;
using std::string;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Resource::_log = spdlog::stderr_color_mt("Resource");


////////////////////
// ctors and dtor //
////////////////////
Resource::~Resource() {
    close();
}


/////////////
// methods //
/////////////
void Resource::open(const Config& cfg, const char* _dir) {
	std::string dir(_dir);

    embed.open(cfg, _dir);
    for (int kernel_size = 2; kernel_size < 6; ++kernel_size) {
        string path = fmt::format("{}/conv.{}.fil", _dir, kernel_size);
        convs[kernel_size].open(path.c_str(), cfg.embed_dim, cfg.embed_dim, kernel_size, &nn::RELU);
    }
    cnv2hdn.open((dir + "/cnv2hdn.lin").c_str(), 4 * cfg.embed_dim, cfg.hidden_dim, true, &nn::RELU);
    string path = fmt::format("{}/hdn2tag.lin", _dir);
    hdn2tag.open(path.c_str(), cfg.hidden_dim, cfg.class_num, true);
    _log->info("NN model loaded");
    preanal.open(_dir);
    errpatch.open(_dir);
    restore.open(_dir);
    _log->info("PoS tagger opened");
}


void Resource::close() {
    embed.close();
    hdn2tag.close();
    preanal.close();
    errpatch.close();
    restore.close();
    _log->debug("PoS tagger closed");
}


}    // namespace khaiii
