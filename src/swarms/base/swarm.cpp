#include "swarm.hpp"
#include "../../common/generator/generator.hpp"
#include "../../common/utils/utils.hpp"
#include <climits>
#include <cmath>
#include <iostream>

template <typename T, typename C>
Swarm<T, C>::Swarm(const base_pso_params_t<T, C> & params):
  b_lo_(params.b_lo),
  b_up_(params.b_up),
  v_min_(-0.2 * (b_up_ - b_lo_)),
  v_max_(-v_min_),
  cog_weight_(params.cog_weight),
  best_idx_(0),
  npar_(params.num_particles),
  ndim_(0),
  prts_(nullptr)
{
}

template <typename T, typename C>
Swarm<T, C>::Swarm(const Swarm<T, C> & other):
  b_lo_(other.b_lo_),
  b_up_(other.b_up_),
  v_min_(other.v_min_),
  v_max_(other.v_max_),
  cog_weight_(other.cog_weight_),
  best_idx_(other.best_idx_),
  npar_(other.npar_),
  ndim_(other.ndim_),
  prts_(other.prts_)
{
}

template <typename T, typename C>
Swarm<T, C> & Swarm<T, C>::operator=(const Swarm<T, C> & other)
{
  if (this != &other)
  {
    this->b_lo_ = other.b_lo_;
    this->b_up_ = other.b_up_;
    this->v_min_ = other.v_min_;
    this->v_max_ = other.v_max_;
    this->cog_weight_ = other.cog_weight_;
    this->best_idx_ = other.best_idx_;
    this->npar_ = other.npar_;
    this->ndim_ = other.ndim_;
    this->prts_ = other.prts_;
  }
  return *this;
}

template <typename T, typename C>
Swarm<T, C>::~Swarm()
{
}

template <typename T, typename C>
void Swarm<T, C>::initializeSwarm()
{
  prts_ = std::make_shared<particles_t<T>>();
  prts_->x.resize(npar_ * ndim_);
  prts_->p.resize(npar_ * ndim_);
  prts_->v.resize(npar_ * ndim_);
  prts_->f_curr.resize(npar_);
  prts_->f_best.resize(npar_);

  for (C i = 0; i < npar_; ++i)
  {
    prts_->f_curr[i] = prts_->f_best[i] = std::numeric_limits<T>::infinity();

    for (C d = 0; d < ndim_; ++d)
    {
      prts_->x[i * ndim_ + d] = prts_->p[i * ndim_ + d] = Generator<T, C>::getReal(b_lo_, b_up_);
      prts_->v[i * ndim_ + d] = Generator<T, C>::getReal(-std::abs(b_up_ - b_lo_), std::abs(b_up_ - b_lo_));
    }
  }
}

template <typename T, typename C>
void Swarm<T, C>::tryToUpdateBestParticle(const C & index)
{
  best_idx_ = better(prts_->f_best[index], prts_->f_best[best_idx_]) ? index : best_idx_;
}

template <typename T, typename C>
typename Vector1D<T>::const_iterator Swarm<T, C>::getStartWeightsIterator(const C & index)
{
  return prts_->x.cbegin() + index * ndim_;
}

template <typename T, typename C>
C Swarm<T, C>::getNumParticles() const
{
  return npar_;
}

template <typename T, typename C>
C Swarm<T, C>::getNumDimensions() const
{
  return ndim_;
}

template <typename T, typename C>
void Swarm<T, C>::setNumDimensions(const C & ndim)
{
  ndim_ = ndim;
}

template <typename T, typename C>
C Swarm<T, C>::getBestIndex()
{
  return best_idx_;
}

template <typename T, typename C>
T Swarm<T, C>::getBlo() const
{
  return b_lo_;
}

template <typename T, typename C>
T Swarm<T, C>::getBup() const
{
  return b_up_;
}

template <typename T, typename C>
T Swarm<T, C>::getCogWeight() const
{
  return cog_weight_;
}
