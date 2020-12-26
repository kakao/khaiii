/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Resource.hpp"


//////////////
// includes //
//////////////
#include <exception>
#include <memory>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

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
void Resource::open(const Config& cfg, std::string dir) {
    embed.open(cfg, dir);
    for (int kernel_size = 2; kernel_size < 6; ++kernel_size) {
        string path = fmt::format("{}/conv.{}.fil", dir, kernel_size);
        convs[kernel_size].open(path, cfg.embed_dim, cfg.embed_dim, kernel_size, &nn::RELU);
    }
    cnv2hdn.open(dir + "/cnv2hdn.lin", 4 * cfg.embed_dim, cfg.hidden_dim, true, &nn::RELU);
    string path = fmt::format("{}/hdn2tag.lin", dir);
    hdn2tag.open(path, cfg.hidden_dim, cfg.class_num, true);
    _log->info("NN model loaded");
    preanal.open(dir);
    errpatch.open(dir);
    restore.open(dir);
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
