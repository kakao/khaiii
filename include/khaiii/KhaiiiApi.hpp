/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#ifndef INCLUDE_KHAIII_KHAIIIAPI_HPP_
#define INCLUDE_KHAIII_KHAIIIAPI_HPP_



//////////////
// includes //
//////////////
#include <exception>
#include <memory>
#include <mutex>    // NOLINT
#include <string>

#include "khaiii/khaiii_api.h"


namespace khaiii {


class KhaiiiApi {
 public:
    /**
     * create khaiii api object
     * @return  shared pointer of khaiii api object
     */
    static std::shared_ptr<KhaiiiApi> create();

    /**
     * open resources
     * @param  rsc_dir  resource directory
     * @param  opt_str  option string (JSON format)
     */
    virtual void open(const char* rsc_dir = "", const char* opt_str = "") = 0;

    /**
     * analyze input text
     * @param  input  input text
     * @param  opt_str  runtime option (JSON format)
     * @return  results
     */
    virtual const khaiii_word_t* analyze(const char* input, const char* opt_str) = 0;

    /**
     * free memories of analyzed results
     * @param  results  results got from analyze() function
     */
    virtual void free_results(const khaiii_word_t* results) = 0;

    virtual void close() = 0;    ///< close resources
};


/**
 * standard exception thrown by khaiii api
 */
class Except: public std::exception {
 public:
    /**
     * @param  msg  error message
     * @param  file  source file (for debug)
     * @param  line  line number in source file (for debug)
     * @param  func  function name (for debug)
     */
    explicit Except(std::string msg, const char* file = nullptr, const int line = 0,
                    const char* func = nullptr);

    virtual const char* what() const noexcept;

    std::string debug();    ///< message with some debug information

 private:
    std::string _msg;    ///< error message
    const char* _file = nullptr;    ///< source file
    const int _line = 0;    ///< line number in source file
    const char* _func = nullptr;    ///< function name
};


}    // namespace khaiii


#endif    // INCLUDE_KHAIII_KHAIIIAPI_HPP_
