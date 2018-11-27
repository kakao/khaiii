/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef INCLUDE_KHAIII_KHAIII_DEV_H_
#define INCLUDE_KHAIII_KHAIII_DEV_H_


//////////////
// includes //
//////////////
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


/**
 * 분석을 수행하고 오분석 패치를 실행하기 직전에 멈춘 다음 그 결과를 리턴한다.
 * @param  handle  handle got from open() function
 * @param  input  input text
 * @param  opt_str  runtime option (JSON format)
 * @param  output  output value for each character
 * @return  output length. -1 if failed
 */
int khaiii_analyze_bfr_errpatch(int handle, const char* input, const char* opt_str,
                                int16_t* output);

/**
 * 로그 레벨을 지정한다.
 * @param  name  로거 이름. "all"인 경우 모든 로거
 * @param  level 로거 레벨. trace, debug, info, warn, err, critical
 * @return  0 if success. -1 if failed
 */
int khaiii_set_log_level(const char* name, const char* level);


/**
 * 여러 로그 레벨을 한꺼번에 지정한다.
 * @param  name_level_pairs  로거 (이름, 레벨) 쌍의 리스트.
 *                           "all:warn,console:info,Tagger:debug"와 같은 형식
 * @return  0 if success. -1 if failed
 */
int khaiii_set_log_levels(const char* name_level_pairs);


#ifdef __cplusplus
}
#endif


#endif    // INCLUDE_KHAIII_KHAIII_DEV_H_
