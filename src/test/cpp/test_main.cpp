/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


//////////////
// includes //
//////////////
#include <iostream>

#include "cxxopts.hpp"
#include "fmt/printf.h"
#ifdef PROFILER
    #include "gperftools/profiler.h"
#endif
#include "gtest/gtest.h"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/stdout_color_sinks.h>

#include "khaiii/khaiii_dev.h"

using std::cerr;
using std::string;




///////////////
// variables //
///////////////
// global variable for program arguments
cxxopts::ParseResult* prog_args;


//////////
// main //
//////////
int main(int argc, char** argv) {
    cxxopts::Options options(argv[0], argv[0]);
    testing::InitGoogleTest(&argc, argv);
    auto _log = spdlog::stderr_color_mt("console");
    spdlog::set_level(spdlog::level::warn);

    options.add_options()
        ("h,help", "print this help")
        ("rsc-dir", "resource directory", cxxopts::value<string>()->default_value("./share/khaiii"))
        ("set-log", "set log level", cxxopts::value<string>()->default_value("all:warn"));
    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        fmt::fprintf(stderr, "%s\n", options.help());
        return 0;
    }
    prog_args = &args;
    khaiii_set_log_levels(args["set-log"].as<string>().c_str());

#ifdef PROFILER
    ProfilerStart("/tmp/test_khaiii.prof");
#endif

    int ret = RUN_ALL_TESTS();

#ifdef PROFILER
    ProfilerStop();
#endif

    return ret;
}
