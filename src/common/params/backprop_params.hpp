#ifndef NN_COMPARISON_BACKPROP_PARAMS_HPP
#define NN_COMPARISON_BACKPROP_PARAMS_HPP

#include <cstddef>

template <typename T, typename C = std::size_t>
struct backprop_params_t
{
  backprop_params_t();
  backprop_params_t(const T & bLow, const T & bUp,
                    const T & learningRate,
                    const T & Momentum,
                    const C & numEpochs);

  T b_lo;
  T b_up;
  T learn_rate;
  T momentum;
  C num_epochs;
};

template struct backprop_params_t<float>;
template struct backprop_params_t<double>;

#endif //NN_COMPARISON_BACKPROP_PARAMS_HPP
