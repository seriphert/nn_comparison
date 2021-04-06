#include "pso_swarm_test.hpp"

template <typename T, typename C>
PSO_Swarm_Test<T, C>::PSO_Swarm_Test(const pso_params_t<T, C> & params):
  PSO_Swarm<T, C>(params)
{
}

template <typename T, typename C>
PSO_Swarm_Test<T, C>::PSO_Swarm_Test(const PSO_Swarm_Test<T, C> & other):
  PSO_Swarm<T, C>(other)
{
}

template <typename T, typename C>
PSO_Swarm_Test<T, C> & PSO_Swarm_Test<T, C>::operator=(const PSO_Swarm_Test<T, C> & other)
{
  PSO_Swarm<T, C>::operator=(other);
  return *this;
}

template <typename T, typename C>
std::shared_ptr<particles_t<T>> & PSO_Swarm_Test<T, C>::getParticles()
{
  return this->prts_;
}
