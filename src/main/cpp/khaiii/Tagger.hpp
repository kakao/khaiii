/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_TAGGER_HPP_
#define SRC_MAIN_CPP_KHAIII_TAGGER_HPP_


//////////////
// includes //
//////////////
#include <vector>
#include <utility>

#include "spdlog/spdlog.h"

#include "khaiii/Embed.hpp"


namespace khaiii {


class Config;
class Resource;
class Sentence;


class Tagger {
 public:
    /**
     * ctor
     * @param  cfg  config
     * @param  rsc  resource
     * @param  sent  Sentence object
     */
    Tagger(const Config& cfg, const Resource& rsc, std::shared_ptr<Sentence> sent);

    void tag();    ///< part-of-speech tag

 private:
    static std::shared_ptr<spdlog::logger> _log;    ///< logger

    const Config& _cfg;    ///< config
    const Resource& _rsc;    ///< resource
    std::shared_ptr<Sentence> _sent;    ///< Sentence object

    /**
     * tag characters with FNN method
     * @param  data  data start point
     * @param  batch_size  batch size
     * @param  col_dim  column dimension for each batch
     */
    void _tag_fnn(float* data, int batch_size, int col_dim,
                  const std::vector<std::pair<int, int>>& index);

    /**
     * tag characters with CNN method
     * @param  data  data start point
     * @param  batch_size  batch size
     * @param  col_dim  column dimension for each batch
     */
    void _tag_cnn(float* data, int batch_size, int col_dim,
                  const std::vector<std::pair<int, int>>& index);

    /**
     * 오분석 패치를 적용하기 전에 예측한 태그를 보정한다.
     * 음절과 태그 조합이 원형복원 사전에 없을 경우 1음절용 태그로 벼환한 다음,
     * B- 위치에 I- 로 잘못 태깅된 태그를 보정한다.
     */
    void _revise_tags();

    void _restore();    ///< restore morphemes

    /**
     * get context embeddings
     */
    std::vector<const embedding_t*> _get_context(int wrd_idx, int chr_idx);

    /**
     * get left context embeddings
     */
    std::vector<const embedding_t*> _get_left_context(int wrd_idx, int chr_idx);

    /**
     * get right context embeddings
     */
    std::vector<const embedding_t*> _get_right_context(int wrd_idx, int chr_idx);
};


}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_TAGGER_HPP_
