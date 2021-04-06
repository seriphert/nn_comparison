#include "iclpso_swarm.hpp"
#include "../../common/generator/generator.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/utils/utils.hpp"
#include <cmath>
#include <algorithm>

template <typename T, typename C>
ICLPSO_Swarm<T, C>::ICLPSO_Swarm(const clpso_params_t<T, C> & params):
  CLPSO_Swarm<T, C>(params),
  num_parts_to_learn_(0),
  desc_idxs_(0)
{
  // Earlier initialization due to regular
  // updates of particles probabilities
  this->pc_.resize(this->npar_);
}

template <typename T, typename C>
ICLPSO_Swarm<T, C>::ICLPSO_Swarm(const ICLPSO_Swarm<T, C> & other):
  CLPSO_Swarm<T, C>(other),
  num_parts_to_learn_(other.num_parts_to_learn_),
  desc_idxs_(other.desc_idxs_)
{
}

template <typename T, typename C>
ICLPSO_Swarm<T, C> & ICLPSO_Swarm<T, C>::operator=(const ICLPSO_Swarm<T, C> & other)
{
  if (this != &other)
  {
    CLPSO_Swarm<T, C>::operator=(other);
    this->num_parts_to_learn_ = other.num_parts_to_learn_;
    this->desc_idxs_ = other.desc_idxs_;
  }
  return *this;
}

template <typename T, typename C>
ICLPSO_Swarm<T, C>::~ICLPSO_Swarm() = default;

template <typename T, typename C>
void ICLPSO_Swarm<T, C>::clearUnusedMemory()
{
  CLPSO_Swarm<T, C>::clearUnusedMemory();
  desc_idxs_.clear();
}

template <typename T, typename C>
void ICLPSO_Swarm<T, C>::generateProbabilities()
{
  // Replace indexes by descending particles fitness value
  std::sort(desc_idxs_.begin(), desc_idxs_.end(),
    [&] (const T & idx1, const T & idx2) {
      return this->prts_->f_best[idx1] > this->prts_->f_best[idx2];
    }
  );

  // Generate Pc[i]
  for (C i = 0; i < this->npar_; ++i)
  {
    this->pc_[desc_idxs_[i]] = (i + 1) / (2. * this->npar_);
  }
}

template <typename T, typename C>
void ICLPSO_Swarm<T, C>::initializeDescendingIndexes()
{
  desc_idxs_.resize(this->npar_);
  std::iota(desc_idxs_.begin(), desc_idxs_.end(), 0);
}

template <typename T, typename C>
bool ICLPSO_Swarm<T, C>::anyParticleNeedsToBeUpdated() const
{
  return std::any_of(this->no_upd_times_.cbegin(), this->no_upd_times_.cend(),
    [&] (C value) {
      return value == this->refreshing_gap_;
    }
  );
}

template <typename T, typename C>
void ICLPSO_Swarm<T, C>::generateNumberOfParticlesForLearn(const C & index,
                                                        const C & epoch)
{
  if (epoch == 0 || this->no_upd_times_[index] >= this->refreshing_gap_)
  {
    C particle_desc_pos = std::distance(desc_idxs_.begin(),
                                      std::find(desc_idxs_.begin(), desc_idxs_.end(), index));
    num_parts_to_learn_ = 2 + std::round((std::ceil(this->npar_ / 2) - 2)
                                    / this->npar_ * (particle_desc_pos + 1));
  }
}

template <typename T, typename C>
void ICLPSO_Swarm<T, C>::chooseParticleForDimension(const C & index,
                                                 const C & dim)
{
  if (Generator<T, C>::getReal(0., 1.) < this->pc_[index])
  {
    // Generate 'num_parts_to_learn_' random unique
    // particles indexes to learn from
    Vector1D<C> idxs
        = generateUniqueIntValuesExceptCurrent<T, C>(num_parts_to_learn_,
                                                     static_cast<C>(0), this->npar_,
                                                     index);

    // Find best particle for train due to its fitness value
    // and remember it for the future learning in d-th dimension
    C p_idx = *std::max_element(idxs.begin(), idxs.end(),
      [&] (const T & idx1, const T & idx2) {
        return better(this->prts_->f_best[idx1], this->prts_->f_best[idx2]);
      }
    );

    this->fi_[index][dim] = p_idx;
  }
  else
  {
    this->fi_[index][dim] = index;
  }
}
