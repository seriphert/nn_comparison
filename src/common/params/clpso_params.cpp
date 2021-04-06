#include "clpso_params.hpp"

template <typename T, typename C>
clpso_params_t<T, C>::clpso_params_t():
  base_pso_params_t<T, C>(),
  refreshing_gap(0)
{
}


template <typename T, typename C>
clpso_params_t<T, C>::clpso_params_t(const C & numParticles,
                                     const T & bLow,
                                     const T & bUp,
                                     const T & cogWeight,
                                     const T & refreshingGap,
                                     const C & numIters,
                                     const std::shared_ptr<InertiaWeight<T>> & weightFunc):
  base_pso_params_t<T, C>(numParticles, bLow, bUp, cogWeight, numIters, weightFunc),
  refreshing_gap(refreshingGap)
{
}

template <typename T, typename C>
clpso_params_t<T, C>::~clpso_params_t() = default;
