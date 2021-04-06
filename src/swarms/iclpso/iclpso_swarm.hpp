#ifndef NN_COMPARISON_ICLPSO_SWARM_HPP
#define NN_COMPARISON_ICLPSO_SWARM_HPP

#include "../clpso/clpso_swarm.hpp"
#include "../../common/params/clpso_params.hpp"
#include "../../common/vector_types.hpp"
#include <cstddef>

template <typename T, typename C = std::size_t>
class ICLPSO_Swarm : public CLPSO_Swarm<T, C>
{
public:
  explicit ICLPSO_Swarm(const clpso_params_t<T, C> & params);
  ICLPSO_Swarm(const ICLPSO_Swarm & other);
  ICLPSO_Swarm(ICLPSO_Swarm && other) = delete;
  ICLPSO_Swarm & operator=(const ICLPSO_Swarm & other);
  ICLPSO_Swarm & operator=(ICLPSO_Swarm && other) = delete;
  ~ICLPSO_Swarm() override;
  void clearUnusedMemory() override;
  void generateProbabilities() override;
  void initializeDescendingIndexes();
  bool anyParticleNeedsToBeUpdated() const;
  void generateNumberOfParticlesForLearn(const C & index,
                                         const C & epoch);

private:
  void chooseParticleForDimension(const C & index,
                                  const C & dim) override;

  // Number of particles to learn
  // for specific particle
  C num_parts_to_learn_;
  // Array of particles indexes in which they placed
  // due to decreasing of their best function values
  Vector1D<C> desc_idxs_;
};

template class ICLPSO_Swarm<float>;
template class ICLPSO_Swarm<double>;

#endif //NN_COMPARISON_ICLPSO_SWARM_HPP
