/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#include "khaiii/khaiii_api.h"


//////////////
// includes //
//////////////
#include <mutex>    // NOLINT
#include <vector>

#include "khaiii/KhaiiiImpl.hpp"


using std::make_shared;
using std::recursive_mutex;
using std::string;
using std::shared_ptr;
using std::unique_lock;
using std::vector;
using khaiii::Except;
using khaiii::KhaiiiApi;
using khaiii::KhaiiiImpl;


///////////////
// variables //
///////////////
/**
 * container for handles. the first (index 0) handle is for special use
 */
vector<shared_ptr<KhaiiiImpl>> KHAIII_HANDLES{ make_shared<KhaiiiImpl>() };


///////////////
// functions //
///////////////
const char* khaiii_version() {
    return KHAIII_VERSION;
}


int khaiii_open(const char* rsc_dir, const char* opt_str) {
    unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
    if (rsc_dir == nullptr) {
        KHAIII_HANDLES[0]->set_err_msg("resource directory is null");
        return -1;
    }
    auto khaiii_impl = make_shared<KhaiiiImpl>();
    try {
        khaiii_impl->open(rsc_dir, opt_str);
        KHAIII_HANDLES.emplace_back(khaiii_impl);
    } catch (const Except& exc) {
        KHAIII_HANDLES[0]->set_err_msg(exc.what());
        return -1;
    }
    return static_cast<int>(KHAIII_HANDLES.size() - 1);
}


const khaiii_word_t* khaiii_analyze(int handle, const char* input, const char* opt_str) {
    if (handle <= 0 || handle >= KHAIII_HANDLES.size()) {
        unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
        KHAIII_HANDLES[0]->set_err_msg(fmt::format("invalid handle: {}", handle).c_str());
        return nullptr;
    }
    auto khaiii_impl = KHAIII_HANDLES[handle];
    if (input == nullptr) {
        khaiii_impl->set_err_msg("input is null");
        return nullptr;
    }
    try {
        return khaiii_impl->analyze(input, opt_str);
    } catch (const Except& exc) {
        khaiii_impl->set_err_msg(exc.what());
        return nullptr;
    }
}


void khaiii_free_results(int handle, const khaiii_word_t* results) {
    unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
    if (handle <= 0 || handle >= KHAIII_HANDLES.size()) {
        KHAIII_HANDLES[0]->set_err_msg(fmt::format("invalid handle: {}", handle).c_str());
        return;
    }
    auto khaiii_impl = KHAIII_HANDLES[handle];
    try {
        khaiii_impl->free_results(results);
    } catch (const Except& exc) {
        khaiii_impl->set_err_msg(exc.what());
    }
}


void khaiii_close(int handle) {
    unique_lock<recursive_mutex> lock(KHAIII_HANDLES[0]->get_mutex());
    if (handle <= 0 || handle >= KHAIII_HANDLES.size()) {
        KHAIII_HANDLES[0]->set_err_msg(fmt::format("invalid handle: {}", handle).c_str());
        return;
    }
    auto khaiii_impl = KHAIII_HANDLES[handle];
    try {
        khaiii_impl->close();
    } catch (const Except& exc) {
        khaiii_impl->set_err_msg(exc.what());
    }
}


const char* khaiii_last_error(int handle) {
    if (handle <= 0 || handle >= KHAIII_HANDLES.size()) handle = 0;
    return KHAIII_HANDLES[handle]->get_err_msg();
}
