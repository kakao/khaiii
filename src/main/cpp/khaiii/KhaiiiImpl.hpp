/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_KHAIIIIMPL_HPP_
#define SRC_MAIN_CPP_KHAIII_KHAIIIIMPL_HPP_


//////////////
// includes //
//////////////
#include <list>
#include <map>
#include <memory>
#include <mutex>    // NOLINT
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "khaiii/Config.hpp"
#include "khaiii/KhaiiiApi.hpp"
#include "khaiii/Resource.hpp"


namespace khaiii {


class Sentence;


/**
 * implementation of khaiii API
 */
class KhaiiiImpl: public KhaiiiApi {
 public:
    virtual ~KhaiiiImpl();    ///< dtor

    void open(const char* rsc_dir = "", const char* opt_str = "") override;

    const khaiii_word_t* analyze(const char* input, const char* opt_str) override;

    /**
     * 분석을 수행하고 오분석 패치를 실행하기 직전에 멈춘 다음 그 결과를 리턴한다.
     * @param  input  input text
     * @param  output  output value for each character
     * @param  opt_str  runtime option (JSON format)
     * @return  output length. -1 if failed
     */
    int analyze_bfr_errpatch(const char* input, const char* opt_str, int16_t* output);

    void free_results(const khaiii_word_t* results) override;

    void close() override;

    /**
     * get mutex for this api object
     * @return  mutex
     */
    std::recursive_mutex& get_mutex();

    /**
     * set error message
     * @param  message
     */
    void set_err_msg(const char* msg);

    /**
     * get error message
     * @return  message
     */
    const char* get_err_msg() const;

    /**
     * 로그 레벨을 지정한다.
     * @param  name  로거 이름. "all"인 경우 모든 로거
     * @param  level 로거 레벨. trace, debug, info, warn, err, critical
     */
    static void set_log_level(const char* name, const char* level);

    /**
     * 여러 로그 레벨을 한꺼번에 지정한다.
     * @param  name_level_pairs  로거 (이름, 레벨) 쌍의 리스트.
     *                           "all:warn,console:info,Tagger:debug"와 같은 형식
     */
    static void set_log_levels(const char* name_level_pairs);


 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    std::recursive_mutex _mutex;    ///< mutex to access exclusively
    bool _is_opened = false;    ///< handle is opened
    std::string _err_msg;    ///< last error message

    Config _cfg;    ///< config
    Resource _rsc;    ///< resource

    // 분석 결과를 C API에 넘겨주고 참조 카운트가 0이 되어 메모리에서 해제되는 것을 방지하기 위해,
    // 헤드 어절을 키로 하여 문장 객체 전체를 임시로 넣어두는 보관소
    std::map<const khaiii_word_t*, std::shared_ptr<Sentence>> _result_cloakroom;

    /**
     * 보관소에 결과를 맡긴다.
     * @param  sent  문장
     * @return  첫번째 어절의 포인터
     */
    const khaiii_word_t* _deposit_sent(std::shared_ptr<Sentence> sent);

    /**
     * 보관하던 결과를 삭제한다.
     * @param  head_word  첫번째 어절의 포인터
     */
    void _withdraw_sent(const khaiii_word_t* head_word);

    /**
     * 리소스 디렉토리를 점검한다.
     * @param  rsc_dir  resource directory
     * @return  존재하는 디렉토리 경로
     */
     std::string _check_rsc_dir(const char* rsc_dir);
};


}    // namespace khaiii


#endif  // SRC_MAIN_CPP_KHAIII_KHAIIIIMPL_HPP_
