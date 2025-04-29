/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_RESOURCE_HPP_
#define SRC_MAIN_CPP_KHAIII_RESOURCE_HPP_


//////////////
// includes //
//////////////
#include <memory>
#include <string>

#include "spdlog/spdlog.h"

#include "khaiii/Embed.hpp"
#include "khaiii/ErrPatch.hpp"
#include "khaiii/Preanal.hpp"
#include "khaiii/Resource.hpp"
#include "khaiii/Restore.hpp"
#include "khaiii/nn/Conv1d.hpp"
#include "khaiii/nn/Linear.hpp"


namespace khaiii {


class Config;


/**
 * resources for part-of-speech tagger
 */
class Resource {
 public:
    virtual ~Resource();    ///< dtor

    Embed embed;    ///< character embedding
    nn::Linear cnv2hdn;    ///< convs -> hidden layer
    nn::Linear hdn2tag;    ///< hidden -> tag(output) layer
    nn::Conv1d convs[6];    ///< convolution layers (0, 1 are dummy)
    Preanal preanal;    ///< 기분석 사전
    ErrPatch errpatch;    ///< 오분석 패치
    Restore restore;    ///< 원형복원 사전

    void open(const Config& cfg, const char* dir);
    void close();

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_RESOURCE_HPP_
