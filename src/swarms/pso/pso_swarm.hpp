#ifndef NN_COMPARISON_PSO_SWARM_HPP
#define NN_COMPARISON_PSO_SWARM_HPP

#include "../base/swarm.hpp"
#include "../../common/params/pso_params.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class PSO_Swarm : public Swarm<T, C>
{
public:
  explicit PSO_Swarm(const pso_params_t<T, C> & params);
  PSO_Swarm(const PSO_Swarm & other);
  PSO_Swarm(PSO_Swarm && other) = delete;
  PSO_Swarm & operator=(const PSO_Swarm & other);
  PSO_Swarm & operator=(PSO_Swarm && other) = delete;
  ~PSO_Swarm() override;
  void updateParticle(const C & index, const T & weight) override;
  bool tryToUpdateBestParticleState(const C & index, const T & func_value) override;
  void clearUnusedMemory() override;
  T getSocWeight() const;

private:
  T soc_weight_;
};

template class PSO_Swarm<float>;
template class PSO_Swarm<double>;

#endif //NN_COMPARISON_PSO_SWARM_HPP
