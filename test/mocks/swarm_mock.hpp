#ifndef NN_COMPARISON_SWARM_MOCK_HPP
#define NN_COMPARISON_SWARM_MOCK_HPP

#include "../../src/swarms/base/swarm.hpp"
#include "../../src/common/params/base_pso_params.hpp"
#include "../../src/swarms/base/particles.hpp"
#include <cstddef>
#include <gmock/gmock.h>

template <typename T>
class SwarmMock : public Swarm<T>
{
public:
  SwarmMock(const base_pso_params_t<T, std::size_t> & params);
  SwarmMock(const SwarmMock & other);
  SwarmMock & operator=(const SwarmMock & other);
  std::shared_ptr<particles_t<T>> & getParticles();

  MOCK_METHOD0_T(clearUnusedMemory, void());
  MOCK_METHOD2_T(updateParticle, void(const std::size_t & index, const T & weight));
  MOCK_METHOD2_T(tryToUpdateBestParticleState, bool(const std::size_t & index, const T & func_value));
};

template class SwarmMock<float>;
template class SwarmMock<double>;

#endif //NN_COMPARISON_SWARM_MOCK_HPP
