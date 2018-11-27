/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_NN_CONV1D_HPP_
#define SRC_MAIN_CPP_KHAIII_NN_CONV1D_HPP_


//////////////
// includes //
//////////////
#include <algorithm>
#include <memory>
#include <string>

#include "khaiii/MemMapFile.hpp"
#include "khaiii/nn/tensor.hpp"


namespace khaiii {
namespace nn {


/**
 * 1D convolution layer
 */
class Conv1d {
 public:
    virtual ~Conv1d();    ///< dtor

    /**
     * open layer parameters
     * @param  path  file path
     * @param  in_ch  input channel
     * @param  out_ch  output channel
     * @param  kernel_size  kernel size
     * @param  activation  activation function
     */
    void open(std::string path, int in_ch, int out_ch, int kernel_size,
              const activation_t* activation = nullptr);

    /**
     * apply forward calculation and also apply max pooling for vector input
     * @param  input  input vector
     * @return  result vector
     */
    vector_t forward_max_pool_vec(const vector_map_t& input) const;

    /**
     * apply forward calculation and also apply max pooling for matrix input
     * @param  input  input matrix. size: [batch size, imput dim]
     * @param  batch_size  batch size
     * @param  col_dim  column dim (for each batch)
     * @return  result matrix
     */
    matrix_t forward_max_pool_mat(const float* data, int batch_size, int col_dim) const;

    void close();    ///< 리소스를 닫는다.

 private:
    std::unique_ptr<matrix_map_t> _weight;    ///< weights [out_ch x (in_ch x kernel)]
    std::unique_ptr<vector_map_t> _bias;    ///< bias [out_ch x 1]
    int _in_ch = 0;    ///< input channel dimension
    int _out_ch = 0;    ///< output chennel dimension
    int _kernel_size = 0;    ///< kernel size
    const activation_t* _activation = nullptr;    ///< activation function

    MemMapFile<float> _param_mmf;    ///< model parameters map file
};


}    // namespace nn
}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_NN_CONV1D_HPP_
