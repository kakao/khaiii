/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#ifndef INCLUDE_KHAIII_KHAIII_API_H_
#define INCLUDE_KHAIII_KHAIII_API_H_


///////////////
// constants //
///////////////
#define KHAIII_VERSION_MAJOR 0
#define KHAIII_VERSION_MINOR 4
#define _MAC2STR(m) #m
#define _JOIN_VER(x,y) _MAC2STR(x) "." _MAC2STR(y)    // NOLINT
#define KHAIII_VERSION _JOIN_VER(KHAIII_VERSION_MAJOR,KHAIII_VERSION_MINOR)    // NOLINT


#ifdef __cplusplus
extern "C" {
#endif


/**
 * morpheme data structure
 */
typedef struct khaiii_morph_t_ {
    const char* lex;    ///< lexical
    const char* tag;    ///< part-of-speech tag
    int begin;    ///< morpheme begin position
    int length;    ///< morpheme length
    char reserved[8];    ///< reserved
    const struct khaiii_morph_t_* next;    ///< next pointer
} khaiii_morph_t;


/**
 * word data structure
 */
typedef struct khaiii_word_t_ {
    int begin;    ///< word begin position
    int length;    ///< word length
    char reserved[8];    ///< reserved
    const khaiii_morph_t* morphs;    ///< morpheme list
    const struct khaiii_word_t_* next;    ///< next pointer
} khaiii_word_t;


/**
 * get version string
 * @return   version string like "2.1"
 */
const char* khaiii_version();


/**
 * open resources
 * @param  rsc_dir  resource directory
 * @param  opt_str  option string (JSON format)
 * @return   handle. -1 if failed
 */
int khaiii_open(const char* rsc_dir, const char* opt_str);


/**
 * analyze input text
 * @param  handle  handle got from open() function
 * @param  input  input text
 * @param  opt_str  runtime option (JSON format)
 * @return  results. NULL if failed
 */
const khaiii_word_t* khaiii_analyze(int handle, const char* input, const char* opt_str);


/**
 * free memories of analyzed results
 * @param  handle  handle got from open() function
 * @param  results  results got from analyze() function
 */
void khaiii_free_results(int handle, const khaiii_word_t* results);


/**
 * close resources
 * @param  handle  handle got from open() function
 */
void khaiii_close(int handle);


/**
 * get last error
 * @param  handle  handle got from open() function
 * @return  message
 */
const char* khaiii_last_error(int handle);


#ifdef __cplusplus
}
#endif


#endif    // INCLUDE_KHAIII_KHAIII_API_H_
