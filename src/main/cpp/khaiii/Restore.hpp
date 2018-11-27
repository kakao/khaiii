/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_RESTORE_HPP_
#define SRC_MAIN_CPP_KHAIII_RESTORE_HPP_


//////////////
// includes //
//////////////
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#include "khaiii/MemMapFile.hpp"
#include "khaiii/Morph.hpp"


namespace khaiii {


/**
 * 원형 복원이 이뤄진 후 음절과 음절별 태그 정보
 */
struct chr_tag_t {
    enum BI { B = 0, I = 1, };    ///< enumeration type for B-, I- notation

    wchar_t chr;
    uint8_t tag;
    BI bi;    ///< B-, I- notation

    inline void set_tag(uint16_t tag_out) {
        assert(tag_out <= 2 * POS_TAG_SIZE);
        tag = tag_out - 1;
        if (tag >= POS_TAG_SIZE) {
            tag -= POS_TAG_SIZE;
            bi = I;
        }
        tag += 1;
    }

    inline void from_val(uint32_t val) {
        chr = val >> 8;    // value의 경우 8비트를 shift하여 음절을 만든다.
        if (val & 0x80) {
            bi = I;
        } else {
            bi = B;
        }
        set_tag(val & 0x7F);
    }

    std::string str();
};


class Restore {
 public:
    virtual ~Restore();    ///< dtor

    /*
     * 리소스를 연다.
     * @param  dir  리소스 디렉토리
     */
    void open(std::string dir);

    void close();    ///< 리소스를 닫는다.

    /**
     * 음절과 그 음절의 태그 번호를 이용해 원형 복원이 필요한 경우 복원한다.
     * @param  chr  음절
     * @param  tag_out  태그 번호
     * @param  use_dic  원형복원 사전을 사용할 지 여부
     * @return   복원한 음절 만큼의 태그 리스트
     */
    std::vector<chr_tag_t> restore(wchar_t chr, uint16_t tag_out, bool use_dic) const;

    /**
     * 원형 복원이 필요한 복합 태그 여부
     * @param  tag_out  태그 번호
     * @return  복합 태그 여부
     */
     static bool is_need_restore(uint16_t tag_out);

    /**
     * 복합 태그가 원형 복원 사전에 존재하는 지 찾는다.
     * @param  chr  음절
     * @param  tag_out  태그 번호
     * @return  인덱스. 찾지 못할 경우 -1
     */
     int find(wchar_t chr, uint16_t tag_out) const;

     /**
      * 원형 복원 사전에 존재하지 않는 복합 태그 번호일 경우 맨 앞에 하나의 태그를 얻는다.
      * @param  tag_out  태그 번호
      * @return  맨 앞에 하나의 태그
      */
     uint8_t get_one(uint16_t tag_out) const;

 private:
    static const int _MAX_VAL_LEN = 4;    ///< maximum array length of value

    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    MemMapFile<uint32_t> _key_mmf;    ///< key memory mapping
    MemMapFile<uint32_t> _val_mmf;    ///< value memory mapping
    MemMapFile<uint8_t> _one_mmf;    ///< one memory mapping

    static int key_cmp(const void* left, const void* right);    ///< key comparator for bsearch
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_RESTORE_HPP_
