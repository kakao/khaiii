/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2017-, Kakao Corp. All rights reserved.
 */


#include "khaiii/nn/tensor.hpp"


//////////////
// includes //
//////////////
#include <algorithm>
#include <functional>
#include <vector>


namespace khaiii {
namespace nn {


using std::vector;


//////////////////////////
// activation functions //
//////////////////////////
float relu(float x) {
    return std::max(x, 0.0f);
}
activation_t RELU = std::ptr_fun(relu);    ///< ReLU function pointer


///////////////
// functions //
///////////////
void add_positional_enc(float* data, int len, int dim) {
    for (int pos = 1; pos <= len; ++pos) {
        float pos_ = pos;
        for (int i = 1; i <= dim; ++i) {
            *data++ += (1.0f - pos_ / len -
                        ((static_cast<float>(i) / dim) * (1.0f - 2.0f * pos_ / len)));
        }
    }
}


}    // namespace nn
}    // namespace khaiii
