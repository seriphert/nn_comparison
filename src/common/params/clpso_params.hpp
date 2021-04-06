#ifndef NN_COMPARISON_CLPSO_PARAMS_HPP
#define NN_COMPARISON_CLPSO_PARAMS_HPP

#include "base_pso_params.hpp"
#include "../inertia_weights.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
struct clpso_params_t : public base_pso_params_t<T, C>
{
  clpso_params_t();
  clpso_params_t(const C & numParticles,
                 const T & bLow,
                 const T & bUp,
                 const T & cogWeight,
                 const T & refreshingGap,
                 const C & numIters,
                 const std::shared_ptr<InertiaWeight<T>> & weightFunc);
  ~clpso_params_t() override;

  C refreshing_gap;
};

template struct clpso_params_t<float>;
template struct clpso_params_t<double>;

#endif //NN_COMPARISON_CLPSO_PARAMS_HPP
