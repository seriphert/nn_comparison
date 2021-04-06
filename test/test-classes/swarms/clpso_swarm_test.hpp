#ifndef NN_COMPARISON_CLPSO_SWARM_TEST_HPP
#define NN_COMPARISON_CLPSO_SWARM_TEST_HPP

#include "../../../src/swarms/clpso/clpso_swarm.hpp"
#include "../../../src/common/params/clpso_params.hpp"
#include "../../../src/swarms/base/particles.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class CLPSO_Swarm_Test: public CLPSO_Swarm<T, C>
{
public:
  CLPSO_Swarm_Test(const clpso_params_t<T, C> & params);
  CLPSO_Swarm_Test(const CLPSO_Swarm_Test<T, C> & other);
  CLPSO_Swarm_Test & operator=(const CLPSO_Swarm_Test<T, C> & other);
  std::shared_ptr<particles_t<T>> & getParticles();
  Vector1D<T> & getProbabilities();
  Vector1D<C> & getNoUpdateTimes();
  Vector2D<C> & getFis();
  bool delegate_between_borders(const std::size_t & index);
  bool delegate_fi_particles_equals_to(const std::size_t & index);
};

template class CLPSO_Swarm_Test<float>;
template class CLPSO_Swarm_Test<double>;

#endif // NN_COMPARISON_CLPSO_SWARM_TEST_HPP
