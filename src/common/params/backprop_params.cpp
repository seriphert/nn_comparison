#include "backprop_params.hpp"
#include <iostream>

template <typename T, typename C>
backprop_params_t<T, C>::backprop_params_t():
  b_lo(0),
  b_up(0),
  learn_rate(0),
  momentum(0),
  num_epochs(0)
{
}

template <typename T, typename C>
backprop_params_t<T, C>::backprop_params_t(const T & bLow, const T & bUp,
                                           const T & learningRate,
                                           const T & Momentum,
                                           const C & numEpochs):
  b_lo(bLow),
  b_up(bUp),
  learn_rate(learningRate),
  momentum(Momentum),
  num_epochs(numEpochs)
{
}
