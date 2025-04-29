/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#include "khaiii/khaiii_dev.h"


//////////////
// includes //
//////////////
#include <vector>
#include <memory>

#include "khaiii/KhaiiiImpl.hpp"


using std::map;
using std::recursive_mutex;
using std::shared_ptr;
using std::string;
using std::unique_lock;
using std::vector;
using khaiii::Except;
using khaiii::KhaiiiImpl;


///////////////
// variables //
///////////////
extern vector<shared_ptr<KhaiiiImpl>> KHAIII_HANDLES;


///////////////
// functions //
///////////////
int khaiii_analyze_bfr_errpatch(int handle, const char* input, const char* opt_str,
                                int16_t* output) {
    if (handle <= 0 || handle >= KHAIII_HANDLES.size()) {
        unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
        KHAIII_HANDLES[0]->set_err_msg(fmt::format("invalid handle: {}", handle).c_str());
        return -1;
    }
    auto khaiii_impl = KHAIII_HANDLES[handle];
    try {
        return khaiii_impl->analyze_bfr_errpatch(input, opt_str, output);
    } catch (const Except& exc) {
        khaiii_impl->set_err_msg(exc.what());
        return -1;
    }
}


int khaiii_set_log_level(const char* name, const char* level) {
    if (name == nullptr || level == nullptr || !name[0] || !level[0]) {
        unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
        KHAIII_HANDLES[0]->set_err_msg("log name or level is null");
        return -1;
    }

    try {
        KhaiiiImpl::set_log_level(name, level);
    } catch (const Except& exc) {
        KHAIII_HANDLES[0]->set_err_msg(exc.what());
        return -1;
    }
    return 0;
}


int khaiii_set_log_levels(const char* name_level_pairs) {
    if (name_level_pairs == nullptr) {
        unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
        KHAIII_HANDLES[0]->set_err_msg("log name/level pair is null");
        return -1;
    }

    try {
        KhaiiiImpl::set_log_levels(name_level_pairs);
    } catch (const Except& exc) {
        KHAIII_HANDLES[0]->set_err_msg(exc.what());
        return -1;
    }
    return 0;
}
