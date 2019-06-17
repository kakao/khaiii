/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_TRIE_HPP_
#define SRC_MAIN_CPP_KHAIII_TRIE_HPP_


//////////////
// includes //
//////////////
#include <functional>
#include <list>
#include <memory>
#include <sstream>
#include <string>

#include "boost/optional.hpp"
#include "spdlog/spdlog.h"

#include "khaiii/MemMapFile.hpp"


namespace khaiii {


/**
 * 유니코드 TRIE
 */
class Trie {
 public:
    struct match_t {    ///< 접두사 매칭 정보를 담기 위한 구조체. common prefix match
        int len;    ///< 매칭된 길이
        uint32_t val;    ///< 값 (양수)
        explicit match_t(int len = -1, uint32_t val = 0): len(len), val(val) {}    ///< ctor
    };

    virtual ~Trie();    ///< dtor

    /**
     * 리소스를 연다.
     * @param  path  파일 경로
     */
    void open(std::string path);

    void close();    ///< 리소스를 닫는다.

    /*
     * 키를 이용해 값을 찾는다.
     * @param  key  키 문자열
     * @return  값. 키가 없을 경우 boost::none
     */
    // boost::optional<uint32_t> find(const std::wstring& key) const;

    /*
     * 키를 이용해 값을 찾는다.
     * @param  key  키 문자열
     * @return  값. 키가 없을 경우 boost::none
     */
    // boost::optional<uint32_t> find(const wchar_t* key) const;

    /*
     * 접두사가 같은 모든 매칭 결과를 검색한다.
     * @param  text  검색할 문자열
     * @return  매칭 결과 리스트
     */
    std::list<match_t> search_common_prefix_matches(const std::wstring& text,
                                                    int max_len = INT_MAX) const;

    /*
     * 접두사가 같은 모든 매칭 결과를 검색한다.
     * @param  text  검색할 문자열
     * @return  매칭 결과 리스트
     */
    std::list<match_t> search_common_prefix_matches(const wchar_t* text,
                                                    int max_len = INT_MAX) const;

    boost::optional<match_t> search_longest_prefix_match(const wchar_t* text,
                                                         int max_len = INT_MAX) const;

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    struct _node_t {    ///< TRIE의 노드 구조체
        wchar_t chr = 0;    ///< 유니코드 문자
        uint32_t val = 0;    ///< 값 (양수). (0인 경우 값이 아님. 즉, 단말 노드가 아님)
        int32_t child_start = -1;    ///< 현재 노드로부터 자식 노드가 시작되는 위치
        int32_t child_num = -1;    ///< 자식 노드의 갯수

        /**
         * 두 노드를 비교하는 함수
         * @param  left  left hand side
         * @param  right  right hand side
         * @return  -1: left < right, 0: left == right, 1: left > right
         */
        static int cmp(const void* left, const void* right) {
            const _node_t* left0 = static_cast<const _node_t*>(left);
            const _node_t* right0 = static_cast<const _node_t*>(right);
            return left0->chr - right0->chr;
        }

        inline std::string str(const _node_t* root_node) const {    ///< 디버그용 문자열 변환
            std::ostringstream oss;
            oss << "node[" << (this - root_node) << "]{'";
            if (chr == 0) {
                oss << "ROOT";
            } else {
                oss << static_cast<char>(chr);
            }
            oss << "', " << val << ", (" << child_start << ", " << child_num << ")}";
            return oss.str();
        }
    };

    MemMapFile<_node_t> _mmf;    ///< memory mapped file

    /*
     * 현재 노드로부터 자식 노드로 내려가며 키 값을 찾는다.
     * @param  key  키 문자열
     * @param  node  노드 시작 위치
     * @return  값. 키가 없을 경우 boost::none
     */
    boost::optional<uint32_t> _find(const wchar_t* key, const _node_t* node) const;

    /*
     * 현재 노드로부터 더이상 매칭되는 키가 없을 때까지 검색한다.
     * @param  text  찾을 텍스트
     * @param  node  노드 시작 위치
     * @param  matches  매칭 결과 리스트
     * @param  len  현재까지 검색을 진행한 길이(자식 노드의 깊이)
     */
    void _search(const wchar_t* text, const _node_t* node, std::list<match_t>* matches,
                 int len, int max_len) const;
};


}    // namespace khaiii


#endif  // SRC_MAIN_CPP_KHAIII_TRIE_HPP_
