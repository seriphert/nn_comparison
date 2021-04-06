#ifndef NN_COMPARISON_BASE_PSO_PARAMS_HPP
#define NN_COMPARISON_BASE_PSO_PARAMS_HPP

#include "../inertia_weights.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
struct base_pso_params_t
{
  base_pso_params_t();
  base_pso_params_t(const C & numParticles, const T & bLow,
                    const T & bUp,
                    const T & cogWeight,
                    const C & numIters,
                    const std::shared_ptr<InertiaWeight<T>> & weightFunc);
  virtual ~base_pso_params_t();

  C num_particles;
  T b_lo;
  T b_up;
  T cog_weight;
  C num_iters;
  std::shared_ptr<InertiaWeight<T>> weight_func;
};

template struct base_pso_params_t<float>;
template struct base_pso_params_t<double>;

#endif //NN_COMPARISON_BASE_PSO_PARAMS_HPP
