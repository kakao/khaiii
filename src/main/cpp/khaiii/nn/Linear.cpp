/**
 * @author  Jamie (jamie.lim@kakaocorp.com)
 * @copyright  Copyright (C) 2018-, Kakao Corp. All rights reserved.
 */


#include "khaiii/nn/Linear.hpp"


//////////////
// includes //
//////////////
#include <string>
#include <vector>

#include "khaiii/util.hpp"


namespace khaiii {
namespace nn {


using std::cout;
using std::endl;
using std::make_unique;
using std::string;
using std::vector;


////////////////////
// ctors and dtor //
////////////////////
Linear::~Linear() {
    close();
}


/////////////
// methods //
/////////////
void Linear::open(string path, int in_dim, int out_dim, bool has_bias,
                  const activation_t* activation) {
    // Eigen은 column 우선으로 저장합니다.
    // 따라서 matrix map의 경우 row, col을 거꾸로 해서 생성한 다음,
    // 사용할 때에는 transpose()를 해서 사용해야 합니다.
    _param_mmf.open(path);
    int size = in_dim * out_dim;
    if (has_bias) size += out_dim;
    assert(_param_mmf.size() == size);
    _weight = make_unique<matrix_map_t>(const_cast<float*>(_param_mmf.data()), in_dim, out_dim);
    if (has_bias) {
        _bias = make_unique<vector_map_t>(const_cast<float*>(_param_mmf.data()) + in_dim * out_dim,
                                          out_dim);
    }
    _activation = activation;
}


/*
#ifndef NDEBUG
    void Linear::print_weight() const {
        int row = _weight->rows();
        int col = _weight->cols();
        fmt::print("============ weight =============\n");
        fmt::print("Size = ({}, {})\n", row, col);
        if (row >= 10 && col >= 10) {
            cout << "first [5 * 5] contents" << endl;
            cout << _weight->block<5, 5>(0, 0) << endl;
            cout << "last [5 * 5] contents" << endl;
            cout << _weight->block<5, 5>(row-5, col-5) << endl;
        } else {
            cout << "contnets" << endl;
            cout << *_weight << endl;
        }
        fmt::print("============ bias =============\n");
        cout << "contnets" << endl;
        cout << _bias->head(5) << endl;
        cout << "..." << endl;
        cout << _bias->tail(5) << endl;
    }
#endif
*/


void Linear::close() {
    _weight.reset();
    _bias.reset();
    _param_mmf.close();
}


}    // namespace nn
}    // namespace khaiii
