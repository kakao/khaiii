/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/nn/Conv1d.hpp"


//////////////
// includes //
//////////////
#include <string>

#include "khaiii/util.hpp"


namespace khaiii {
namespace nn {


using std::make_unique;
using std::string;


////////////////////
// ctors and dtor //
////////////////////
Conv1d::~Conv1d() {
    close();
}


/////////////
// methods //
/////////////
void Conv1d::open(string path, int in_ch, int out_ch, int kernel_size,
                  const activation_t* activation) {
    _param_mmf.open(path);
    assert(_param_mmf.size() == (in_ch * out_ch * kernel_size + out_ch));
    // [output channel * [kernel * input channel]] ==> transposed
    // ==> [[kernel * input channel] * output channel]
    // 즉, 저장은 [row, col]으로 했지만 사용은 [col, row]로 접근해야 합니다.
    _weight = make_unique<matrix_map_t>(const_cast<float*>(_param_mmf.data()), kernel_size * in_ch,
                                        out_ch);
    _bias = make_unique<vector_map_t>(const_cast<float*>(_param_mmf.data()) + \
                                      (in_ch * out_ch * kernel_size), out_ch);
    _in_ch = in_ch;
    _out_ch = out_ch;
    _kernel_size = kernel_size;
    _activation = activation;
}


vector_t Conv1d::forward_max_pool_vec(const vector_map_t& input) const {
    int out_row_size = (input.size() / _in_ch) - (_kernel_size - 1);
    int in_col_size = _in_ch * _kernel_size;
    matrix_t output(out_row_size, _out_ch);
    for (int row = 0; row < out_row_size; ++row) {
        output.row(row) = _weight->transpose() * input.segment(row * _in_ch, in_col_size) + *_bias;
    }
    auto pooled = output.colwise().maxCoeff();
    if (_activation) return pooled.unaryExpr(*_activation);
    return pooled;
}


matrix_t Conv1d::forward_max_pool_mat(const float* data, int batch_size, int col_dim) const {
    matrix_t outputs(batch_size, _out_ch);
    for (int batch = 0; batch < batch_size; ++batch) {
        vector_map_t vec(const_cast<float*>(data + batch * col_dim), col_dim);
        outputs.row(batch) = forward_max_pool_vec(vec);
    }
    return outputs;
}


void Conv1d::close() {
    _weight.release();
    _bias.release();
    _param_mmf.close();
}


}    // namespace nn
}    // namespace khaiii
