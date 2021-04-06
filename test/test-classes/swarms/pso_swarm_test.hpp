#ifndef NN_COMPARISON_PSO_SWARM_TEST_HPP
#define NN_COMPARISON_PSO_SWARM_TEST_HPP

#include "../../../src/swarms/base/particles.hpp"
#include "../../../src/swarms/pso/pso_swarm.hpp"
#include "../../../src/common/params/pso_params.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class PSO_Swarm_Test: public PSO_Swarm<T, C>
{
public:
  PSO_Swarm_Test(const pso_params_t<T, C> & params);
  PSO_Swarm_Test(const PSO_Swarm_Test<T, C> & other);
  PSO_Swarm_Test & operator=(const PSO_Swarm_Test<T, C> & other);
  std::shared_ptr<particles_t<T>> & getParticles();
};

template class PSO_Swarm_Test<float>;
template class PSO_Swarm_Test<double>;

#endif // NN_COMPARISON_PSO_SWARM_TEST_HPP
