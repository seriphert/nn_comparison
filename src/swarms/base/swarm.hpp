#ifndef NN_COMPARISON_SWARM_HPP
#define NN_COMPARISON_SWARM_HPP

#include "../../common/params/base_pso_params.hpp"
#include "../../common/vector_types.hpp"
#include "particles.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class Swarm
{
public:
  explicit Swarm(const base_pso_params_t<T, C> & params);
  Swarm(const Swarm & other);
  Swarm(Swarm && other) = delete;
  Swarm & operator=(const Swarm & other);
  Swarm & operator=(Swarm && other) = delete;
  virtual ~Swarm() = 0;
  void initializeSwarm();
  virtual void updateParticle(const C & index, const T & weight) = 0;
  virtual bool tryToUpdateBestParticleState(const C & index, const T & func_value) = 0;
  void tryToUpdateBestParticle(const C & index);
  virtual void clearUnusedMemory() = 0;
  typename Vector1D<T>::const_iterator getStartWeightsIterator(const C & index);
  C getNumParticles() const;
  C getNumDimensions() const;
  C getBestIndex();
  void setNumDimensions(const C & ndim);
  T getBlo() const;
  T getBup() const;
  T getCogWeight() const;

protected:
  T b_lo_;
  T b_up_;
  T v_min_;
  T v_max_;
  T cog_weight_;
  C npar_;
  C ndim_;
  C best_idx_;
  std::shared_ptr<particles_t<T>> prts_;
};

template class Swarm<float>;
template class Swarm<double>;

#endif //NN_COMPARISON_SWARM_HPP
