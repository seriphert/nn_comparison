#include "swarm_mock.hpp"
#include <iostream>

template <typename T>
SwarmMock<T>::SwarmMock(const base_pso_params_t<T, std::size_t> & params):
  Swarm<T>(params)
{
}

template <typename T>
SwarmMock<T>::SwarmMock(const SwarmMock<T> & other):
  Swarm<T>(other)
{
}

template <typename T>
SwarmMock<T> & SwarmMock<T>::operator=(const SwarmMock<T> & other)
{
  Swarm<T>::operator=(other);
  return *this;
}

template <typename T>
std::shared_ptr<particles_t<T>> & SwarmMock<T>::getParticles()
{
  return this->prts_;
}
