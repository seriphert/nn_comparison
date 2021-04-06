#include "clpso_swarm_test.hpp"

template <typename T, typename C>
CLPSO_Swarm_Test<T, C>::CLPSO_Swarm_Test(const clpso_params_t<T, C> & params):
  CLPSO_Swarm<T, C>(params)
{
}

template <typename T, typename C>
CLPSO_Swarm_Test<T, C>::CLPSO_Swarm_Test(const CLPSO_Swarm_Test<T, C> & other):
  CLPSO_Swarm<T, C>(other)
{
}

template <typename T, typename C>
CLPSO_Swarm_Test<T, C> & CLPSO_Swarm_Test<T, C>::operator=(const CLPSO_Swarm_Test<T, C> & other)
{
  CLPSO_Swarm<T, C>::operator=(other);
  return *this;
}

template <typename T, typename C>
std::shared_ptr<particles_t<T>> & CLPSO_Swarm_Test<T, C>::getParticles()
{
  return this->prts_;
}

template <typename T, typename C>
Vector1D<T> & CLPSO_Swarm_Test<T, C>::getProbabilities()
{
  return this->pc_;
}

template <typename T, typename C>
Vector1D<C> & CLPSO_Swarm_Test<T, C>::getNoUpdateTimes()
{
  return this->no_upd_times_;
}

template <typename T, typename C>
Vector2D<C> & CLPSO_Swarm_Test<T, C>::getFis()
{
  return this->fi_;
}

template <typename T, typename C>
bool CLPSO_Swarm_Test<T, C>::delegate_between_borders(const std::size_t & index)
{
  return this->betweenBorders(index);
}

template <typename T, typename C>
bool CLPSO_Swarm_Test<T, C>::delegate_fi_particles_equals_to(const std::size_t & index)
{
  return this->allFiPartsEqualTo(index);
}
