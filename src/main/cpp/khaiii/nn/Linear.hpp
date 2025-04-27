/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_NN_LINEAR_HPP_
#define SRC_MAIN_CPP_KHAIII_NN_LINEAR_HPP_


//////////////
// includes //
//////////////
#include <iostream>
#include <algorithm>
#include <memory>
#include <string>

#include "khaiii/MemMapFile.hpp"
#include "khaiii/nn/tensor.hpp"
#include "spdlog/spdlog.h"
#include "fmt/format.h"


namespace khaiii {
namespace nn {


/**
 * fully connected layer
 */
class Linear {
 public:
    virtual ~Linear();
    /**
     * open layer parameters
     * @param  path  file path
     * @param  in_dim  input dimension
     * @param  out_dim   output dimension
     * @param  has_bias  whether has bias or not
     * @param  activation  activation function
     */
    void open(const char* path, int in_dim, int out_dim, bool has_bias,
              const activation_t* activation = nullptr);

    void close();    ///< 리소스를 닫는다.

    /**
     * apply forward calculation for vector input
     * @param  input  input vector
     * @return  result vector
     */
    template<typename T>
    inline vector_t forward_vec(const T &input) const {
        auto without_bias = _weight->transpose() * input;
        if (_bias.get() == nullptr) {
            if (_activation) return without_bias.unaryExpr(*_activation);
            return without_bias;
        }
        auto with_bias = without_bias + *_bias;
        if (_activation) return with_bias.unaryExpr(*_activation);
        return with_bias;
    }

    /**
     * apply forward calculation for matrix input
     * @param  input  input matrix. size: [batch size, input dim]
     * @return  result matrix
     */
    template<typename T>
    inline matrix_t forward_mat(const T& input) const {
        auto without_bias = input * *_weight;
        if (_bias.get() == nullptr) {
            if (_activation) return without_bias.unaryExpr(*_activation);
            return without_bias;
        }
        auto with_bias = without_bias.transpose().colwise() + *_bias;
        if (_activation) return with_bias.unaryExpr(*_activation).transpose();
        return with_bias.transpose();
    }

    /*
    #ifndef NDEBUG
        void print_weight() const;    ///< print weights for debugging
    #endif
    */

 private:
    std::unique_ptr<matrix_map_t> _weight;    ///< weights [out x in]
    std::unique_ptr<vector_map_t> _bias;    ///< bias [out x 1]
    const activation_t* _activation = nullptr;    ///< activation function

    MemMapFile<float> _param_mmf;    ///< model parameters map file
};


}    // namespace nn
}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_NN_LINEAR_HPP_
