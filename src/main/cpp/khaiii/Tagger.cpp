/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/Tagger.hpp"


//////////////
// includes //
//////////////

#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "khaiii/Config.hpp"
#include "khaiii/Embed.hpp"
#include "khaiii/Sentence.hpp"
#include "khaiii/Word.hpp"
#include "khaiii/util.hpp"
#include "khaiii/nn/tensor.hpp"

/** Supports spdlog::stderr_color_mt */
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace khaiii {


using std::copy;
using std::dynamic_pointer_cast;
using std::make_pair;
using std::pair;
using std::shared_ptr;
using std::vector;


////////////////////
// static members //
////////////////////
shared_ptr<spdlog::logger> Tagger::_log = spdlog::stderr_color_mt("Tagger");


////////////////////
// ctors and dtor //
////////////////////
Tagger::Tagger(const Config& cfg, const Resource& rsc, shared_ptr<Sentence> sent):
        _cfg(cfg), _rsc(rsc), _sent(sent) {}


/////////////
// methods //
/////////////
void Tagger::tag() {
    for (auto& word : _sent->words) {
        if (_cfg.preanal) _rsc.preanal.apply(word);
        word->set_embeds(_rsc);
    }

    vector<vector<const embedding_t*>> batch;
    vector<pair<int, int>> index;    // (word, char) index
    for (int i = 0; i < _sent->words.size(); ++i) {
        auto word = _sent->words[i];
        for (int j = 0; j < word->wlength; ++j) {
            if (word->char_tags[j] > 0) continue;    // 이미 기분석 사전이 적용된 경우
            batch.emplace_back(_get_context(i, j));
            index.emplace_back(make_pair(i, j));
        }
    }

    int col_dim = (_cfg.window * 2 + 1) * _cfg.embed_dim;
    vector<float> data(col_dim * batch.size());
    for (int i = 0; i < batch.size(); ++i) {
        for (int j = 0; j < batch[i].size(); ++j) {
            copy(batch[i][j]->data(), batch[i][j]->data() + _cfg.embed_dim,
                 &data[i * col_dim + j * _cfg.embed_dim]);
        }
        _add_lwb_rwb(&data[i * col_dim], index[i].first, index[i].second);
        nn::add_positional_enc(&data[i * col_dim], _cfg.window * 2 + 1, _cfg.embed_dim);
#ifndef NDEBUG
        for (int j = 0; j < batch[i].size(); ++j) {
            SPDLOG_TRACE(_log, "batch[{}][{}]", i, j);
            for (int k = 0; k < _cfg.embed_dim; ++k) {
                SPDLOG_TRACE(_log, "\t{}: {}", k, data[i * col_dim + j * _cfg.embed_dim + k]);
            }
        }
#endif
    }

    _tag_cnn(data.data(), batch.size(), col_dim, index);

    _revise_tags();
    if (_cfg.errpatch) _rsc.errpatch.apply(_sent);    // 오분석 패치
    _restore();    // 원형 복원
}


void Tagger::_add_lwb_rwb(float* data, int wrd_idx, int chr_idx) {
    int context_len = _cfg.window * 2 + 1;
    int lwb_idx = context_len / 2 + _sent->get_lwb_delta(wrd_idx, chr_idx);
    if (lwb_idx >= 0) {
        nn::add_vec(&data[lwb_idx * _cfg.embed_dim], _rsc.embed.left_word_bound().data(),
                    _cfg.embed_dim);
    }
    int rwb_idx = context_len / 2 + _sent->get_rwb_delta(wrd_idx, chr_idx);
    if (rwb_idx < context_len) {
        nn::add_vec(&data[rwb_idx * _cfg.embed_dim], _rsc.embed.right_word_bound().data(),
                    _cfg.embed_dim);
    }
}


void Tagger::_tag_cnn(float* data, int batch_size, int col_dim,
                      const vector<pair<int, int>>& index) {
    nn::matrix_t features(batch_size, 4 * _cfg.embed_dim);
    features << _rsc.convs[2].forward_max_pool_mat(data, batch_size, col_dim),
                 _rsc.convs[3].forward_max_pool_mat(data, batch_size, col_dim),
                 _rsc.convs[4].forward_max_pool_mat(data, batch_size, col_dim),
                 _rsc.convs[5].forward_max_pool_mat(data, batch_size, col_dim);
    auto hidden_outs = _rsc.cnv2hdn.forward_mat(features);
    auto logits = _rsc.hdn2tag.forward_mat(hidden_outs);
    for (int i = 0; i < batch_size; ++i) {
        nn::vector_t::Index max_idx;
        logits.row(i).maxCoeff(&max_idx);
        int wrd_idx = index[i].first;
        int chr_idx = index[i].second;
        _sent->words[wrd_idx]->char_tags[chr_idx] = max_idx + 1;
    }
}


void Tagger::_revise_tags() {
    for (int i = 0; i < _sent->words.size(); ++i) {
        auto word = _sent->words[i];
        uint16_t prev_tag = 0;
        for (int j = 0; j < word->wlength; ++j) {
            uint16_t curr_tag = word->char_tags[j];
            if (Restore::is_need_restore(curr_tag) &&
                    _rsc.restore.find(word->wbegin[j], curr_tag) == -1) {
                // 원형 복원이 필요하지만 사전에는 존재하지 않는 경우,
                // 복합 태그의 첫번째 태그로 치환해준다.
                auto restored = _rsc.restore.restore(word->wbegin[j], curr_tag, false);
                assert(restored.size() == 1);
                curr_tag = restored[0].tag;
                if (restored[0].bi == chr_tag_t::I) curr_tag += POS_TAG_SIZE;
                _log->debug("tag one: {} -> {}", word->char_tags[j], curr_tag);
                word->char_tags[j] = curr_tag;
            }
            if (0 < curr_tag && curr_tag <= POS_TAG_SIZE) {
                // B- 태그이면서 이전 카테고리와 다른 경우 I- 태그로 보정해준다.
                if (j == 0 || !_is_same_tag_cat(word->wbegin[j-1], prev_tag, curr_tag)) {
                    curr_tag += POS_TAG_SIZE;
                    _log->debug("B->I tag: {} -> {}", word->char_tags[j], curr_tag);
                    word->char_tags[j] = curr_tag;
                }
            }
            prev_tag = curr_tag;
        }
    }
}


bool Tagger::_is_same_tag_cat(char32_t prev_chr, int prev_tag, int curr_tag) {
    assert(0 < curr_tag && curr_tag <= POS_TAG_SIZE);
    if (prev_tag == 0) return false;    // 맨 첫번째 음절인 경우 항상 false
    if (0 < prev_tag && prev_tag <= 2 * POS_TAG_SIZE) {
        // 이전 태그가 단순 태그일 경우
        return (prev_tag-1) % POS_TAG_SIZE == (curr_tag-1) % POS_TAG_SIZE;
    }
    // 이전 태그가 복합 태그일 경우 원형복원 후 마지막 음절의 태그로 판단한다.
    auto restored = _rsc.restore.restore(prev_chr, prev_tag, true);
    int prev_last_tag = restored[restored.size()-1].tag;
    return (prev_last_tag-1) % POS_TAG_SIZE == (curr_tag-1) % POS_TAG_SIZE;
}


void Tagger::_restore() {
    for (int i = 0; i < _sent->words.size(); ++i) {
        auto word = _sent->words[i];
        for (int j = 0; j < word->wlength; ++j) {
            auto restored = _rsc.restore.restore(word->wbegin[j], word->char_tags[j], _cfg.restore);
            word->restored.emplace_back(restored);
#ifndef NDEBUG
            for (auto& chr_tag : word->restored[j]) {
                _log->debug("\t{}", chr_tag.str());
            }
#endif
        }
        word->make_morphs();
    }
}


vector<const embedding_t*> Tagger::_get_context(int wrd_idx, int chr_idx) {
    vector<const embedding_t*> context;
    context.reserve(_cfg.window * 2 + 1);
    auto left_context = _get_left_context(wrd_idx, chr_idx);
    context.insert(context.end(), left_context.rbegin(), left_context.rend());
    context.emplace_back(&_sent->words[wrd_idx]->embeds[chr_idx]);    // current character itself
    auto right_context = _get_right_context(wrd_idx, chr_idx);
    context.insert(context.end(), right_context.begin(), right_context.end());
    assert(context.size() == _cfg.window * 2 + 1);
    return context;
}


vector<const embedding_t*> Tagger::_get_left_context(int wrd_idx, int chr_idx) {
    vector<const embedding_t*> left_context;
    left_context.reserve(_cfg.window);
    // current word, from left character
    auto word = _sent->words[wrd_idx];
    for (int c = chr_idx - 1; c >= 0 && left_context.size() < _cfg.window; --c) {
        left_context.emplace_back(&(word->embeds.at(c)));
    }
    // from left word
    for (int w = wrd_idx - 1; w >= 0 && left_context.size() < _cfg.window; --w) {
        word = _sent->words[w];
        for (int c = word->wlength - 1; c >= 0 && left_context.size() < _cfg.window; --c) {
            left_context.emplace_back(&(word->embeds.at(c)));
        }
    }
    // left padding
    while (left_context.size() < _cfg.window) {
        left_context.emplace_back(&_rsc.embed.left_padding());
    }
    assert(left_context.size() == _cfg.window);
    return left_context;
}


vector<const embedding_t*> Tagger::_get_right_context(int wrd_idx, int chr_idx) {
    vector<const embedding_t*> right_context;
    right_context.reserve(_cfg.window);
    // current word, from right character
    auto word = _sent->words[wrd_idx];
    for (int c = chr_idx + 1; c < word->wlength && right_context.size() < _cfg.window; ++c) {
        right_context.emplace_back(&(word->embeds.at(c)));
    }
    // from right word
    for (int w = wrd_idx + 1; w < _sent->words.size() && right_context.size() < _cfg.window; ++w) {
        word = _sent->words[w];
        for (int c = 0; c < word->wlength && right_context.size() < _cfg.window; ++c) {
            right_context.emplace_back(&(word->embeds.at(c)));
        }
    }
    // right padding
    while (right_context.size() < _cfg.window) {
        right_context.emplace_back(&_rsc.embed.right_padding());
    }
    assert(right_context.size() == _cfg.window);
    return right_context;
}


}    // namespace khaiii
