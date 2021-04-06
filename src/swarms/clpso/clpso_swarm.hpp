#ifndef NN_COMPARISON_CLPSO_SWARM_HPP
#define NN_COMPARISON_CLPSO_SWARM_HPP

#include "../base/swarm.hpp"
#include "../../common/params/clpso_params.hpp"
#include "../../common/vector_types.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class CLPSO_Swarm : public Swarm<T, C>
{
public:
  explicit CLPSO_Swarm(const clpso_params_t<T, C> & params);
  CLPSO_Swarm(const CLPSO_Swarm & other);
  CLPSO_Swarm(CLPSO_Swarm && other) = delete;
  CLPSO_Swarm & operator=(const CLPSO_Swarm & other);
  CLPSO_Swarm & operator=(CLPSO_Swarm && other) = delete;
  ~CLPSO_Swarm() override;
  void updateParticle(const C & index, const T & weight) override;
  bool tryToUpdateBestParticleState(const C & index,
                                    const T & func_value) override;
  void clearUnusedMemory() override;
  void initializeFiArray();
  void initializeNoUpdateTimes();
  virtual void generateProbabilities();
  void chooseParticlesToLearnFrom(const C & index, const C & epoch);
  T getRefreshingGap() const;

protected:
  bool betweenBorders(const C & index) const;
  bool allFiPartsEqualTo(const C & index) const;
  virtual void chooseParticleForDimension(const C & index, const C & dim);

  // Particles probabilities
  Vector1D<T> pc_;
  // Maximal time with no updates
  C refreshing_gap_;
  // Particles no update times array
  Vector1D<C> no_upd_times_;
  // Particles to learn from in each dimension for each particle
  Vector2D<C> fi_;
};

template class CLPSO_Swarm<float>;
template class CLPSO_Swarm<double>;

#endif //NN_COMPARISON_CLPSO_SWARM_HPP
