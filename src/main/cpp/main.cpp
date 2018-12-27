/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


//////////////
// includes //
//////////////
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

#include "cxxopts.hpp"
#include "fmt/printf.h"
#ifdef PROFILER
    #include "gperftools/profiler.h"
#endif
#include "spdlog/spdlog.h"

#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/khaiii_dev.h"


using std::cerr;
using std::cin;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;

using khaiii::KhaiiiApi;


///////////////
// functions //
///////////////
int run(const cxxopts::ParseResult& opts) {
    auto _log = spdlog::get("console");
    khaiii_set_log_levels(opts["set-log"].as<string>().c_str());

    auto khaiii_api = KhaiiiApi::create();
    try {
        khaiii_api->open(opts["rsc-dir"].as<string>(), opts["opt-str"].as<string>());
    } catch (const khaiii::Except& exc) {
        _log->error("fail to open dir: '{}', opt: '{}'", opts["rsc-dir"].as<string>(),
                    opts["opt-str"].as<string>());
        _log->error(exc.what());
        return 1;
    }

    for (string line; getline(cin, line); ) {
        _log->debug("sent: {}", line);
        const khaiii_word_t* results = nullptr;
        try {
            results = khaiii_api->analyze(line.c_str(), "");
        } catch (const khaiii::Except& exc) {
            _log->warn("{}: {}", exc.what(), line);
            continue;
        }
        for (auto word = results; word != nullptr; word = word->next) {
            fmt::print("{}\t", line.substr(word->begin, word->length));
            const khaiii_morph_t* morphs = word->morphs;
            for (auto morph = morphs; morph != nullptr; morph = morph->next) {
                if (morph != morphs) fmt::print(" + ");
                fmt::print("{}/{}", morph->lex, morph->tag);
            }
            fmt::print("\n");
        }
        fmt::print("\n");
        khaiii_api->free_results(results);
    }

    return 0;
}


//////////
// main //
//////////
int main(int argc, char** argv) {
    auto _log = spdlog::stderr_color_mt("console");
    spdlog::set_level(spdlog::level::warn);

    cxxopts::Options options("khaiii", "analyze with khaiii");
    options.add_options()
        ("h,help", "print this help")
        ("rsc-dir", "resource directory", cxxopts::value<string>()->default_value(""))
        ("opt-str", "option (JSON format)", cxxopts::value<string>()->default_value(""))
        ("input", "input file (default: stdin)", cxxopts::value<string>())
        ("output", "output file (default: stdout)", cxxopts::value<string>())
        ("set-log", "set log level", cxxopts::value<string>()->default_value("all:info"));
    auto opts = options.parse(argc, argv);

    if (opts.count("help")) {
        fmt::fprintf(cerr, "%s\n", options.help());
        return 0;
    }
    if (opts.count("input")) {
        string path = opts["input"].as<string>();
        ifstream fin(path);
        if (!fin.good()) {
            _log->error("input file not found: {}", path);
            return 1;
        }
        if (freopen(path.c_str(), "r", stdin) == nullptr) {
            _log->error("fail to open input file: {}", path);
            return 2;
        }
    }
    if (opts.count("output")) {
        string path = opts["output"].as<string>();
        if (freopen(path.c_str(), "w", stdout) == nullptr) {
            _log->error("fail to open output file: {}", path);
            return 3;
        }
    }

#ifdef PROFILER
    ProfilerStart("/tmp/bin_khaiii.prof");
#endif

    int ret = run(opts);

#ifdef PROFILER
    ProfilerStop();
#endif

    return ret;
}
