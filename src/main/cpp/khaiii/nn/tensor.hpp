/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#ifndef SRC_MAIN_CPP_KHAIII_NN_TENSOR_HPP_
#define SRC_MAIN_CPP_KHAIII_NN_TENSOR_HPP_


//////////////
// includes //
//////////////
#include <functional>

#include "Eigen/Dense"


namespace khaiii {
namespace nn {


///////////
// types //
///////////
using matrix_map_t = Eigen::Map<Eigen::MatrixXf>;
using vector_map_t = Eigen::Map<Eigen::VectorXf>;
using matrix_t = Eigen::MatrixXf;
using vector_t = Eigen::VectorXf;


//////////////////////////
// activation functions //
//////////////////////////
typedef std::pointer_to_unary_function<float, float> activation_t;
extern activation_t RELU;


///////////////
// functions //
///////////////
/**
 * add positional encoding to data(array of floats)
 * @param  data  input data. size: [length x dimension]
 * @param  len  position length
 * @param  dim  embedding dimension
 */
void add_positional_enc(float* data, int len, int dim);

/**
 * add two vector in-place (update left vector)
 * @param  left  vector (will be updated)
 * @param  right  vector
 */
inline void add_vec(float* left, const float* right, int dim) {
    assert(dim > 0);
    for (; dim > 0; --dim) *left++ += *right++;
}


}    // namespace nn
}    // namespace khaiii


#endif    // SRC_MAIN_CPP_KHAIII_NN_TENSOR_HPP_
