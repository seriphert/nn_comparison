/*
#ifndef NN_COMPARISON_ICLPSO_SWARM_TEST_HPP
#define NN_COMPARISON_ICLPSO_SWARM_TEST_HPP

#include "../../../src/swarms/iclpso/iclpso_swarm.hpp"
#include "../../../src/swarms/base/particles.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class ICLPSO_Swarm_Test: public ICLPSO_Swarm<T, C>
{
public:
  std::shared_ptr<particles_t<T>> & getParticles();
};

template class ICLPSO_Swarm_Test<float>;
template class ICLPSO_Swarm_Test<double>;

#endif // NN_COMPARISON_ICLPSO_SWARM_TEST_HPP
*/