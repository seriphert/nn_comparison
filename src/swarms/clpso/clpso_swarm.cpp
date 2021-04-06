#include "clpso_swarm.hpp"
#include "../../common/generator/generator.hpp"
#include "../../common/utils/utils.hpp"
#include <cmath>
#include <algorithm>

template <typename T, typename C>
CLPSO_Swarm<T, C>::CLPSO_Swarm(const clpso_params_t<T, C> & params):
  Swarm<T, C>(params),
  pc_(0),
  refreshing_gap_(params.refreshing_gap),
  no_upd_times_(0),
  fi_(0)
{
}

template <typename T, typename C>
CLPSO_Swarm<T, C>::CLPSO_Swarm(const CLPSO_Swarm<T, C> & other):
  Swarm<T, C>(other),
  pc_(other.pc_),
  refreshing_gap_(other.refreshing_gap_),
  no_upd_times_(other.no_upd_times_),
  fi_(other.fi_)
{
}

template <typename T, typename C>
CLPSO_Swarm<T, C> & CLPSO_Swarm<T, C>::operator=(const CLPSO_Swarm<T, C> & other)
{
  if (this != &other)
  {
    Swarm<T, C>::operator=(other);
    this->pc_ = other.pc_;
    this->refreshing_gap_ = other.refreshing_gap_;
    this->no_upd_times_ = other.no_upd_times_;
    this->fi_ = other.fi_; 
  }
  return *this;
}

template <typename T, typename C>
CLPSO_Swarm<T, C>::~CLPSO_Swarm() = default;

template <typename T, typename C>
void CLPSO_Swarm<T, C>::updateParticle(const C & index, const T & weight)
{
  C dim = this->ndim_;

  for (C d = 0; d < dim; ++d)
  {
    T r = Generator<T, C>::getReal(0., 1.);

    T temp = weight * this->prts_->v[index * dim + d]
            + this->cog_weight_ * r * (this->prts_->p[fi_[index][d] * dim + d]
            - this->prts_->x[index * dim + d]);

    temp = (std::abs(temp) > this->v_max_)
          ? ((temp > this->v_max_) ? this->v_max_: this->v_min_)
          : temp;

    this->prts_->x[index * dim + d] += (this->prts_->v[index * dim + d] = temp);
  }
}

template <typename T, typename C>
bool CLPSO_Swarm<T, C>::tryToUpdateBestParticleState(const C & index, const T & func_value)
{
  C dim = this->ndim_;
  this->prts_->f_curr[index] = func_value;

  if (betweenBorders(index))
  {
    if (better(this->prts_->f_curr[index], this->prts_->f_best[index]))
    {
      no_upd_times_[index] = 0;

      this->prts_->f_best[index] = this->prts_->f_curr[index];
      std::copy(&this->prts_->x[index * dim], &this->prts_->x[index * dim + dim],
                  &this->prts_->p[index * dim]);

      // Successful particle's best value update, return 'true'
      return true;
    }
    else
    {
      ++no_upd_times_[index];
    }
  }

  // Unsuccessful particle's best value update, return 'false'
  return false;
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::clearUnusedMemory()
{
  this->prts_.reset();
  this->prts_ = nullptr;
  pc_.clear();
  fi_.clear();
  no_upd_times_.clear();
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::initializeFiArray()
{
  fi_.resize(this->npar_);

  for (C i = 0; i < this->npar_; ++i)
  {
    fi_[i].resize(this->ndim_);
  }
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::initializeNoUpdateTimes()
{
  no_upd_times_.assign(this->npar_, refreshing_gap_);
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::generateProbabilities()
{
  pc_.resize(this->npar_);

  for (C i = 0; i < this->npar_; ++i)
  {
    pc_[i] = 0.05 + 0.45 * (std::expm1(10 * static_cast<T>(i + 1) / this->npar_))
            / (std::expm1(10));
  }
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::chooseParticlesToLearnFrom(const C & index, const C & epoch)
{
  if (epoch == 0 || no_upd_times_[index] >= refreshing_gap_)
  {
    // Select particles for each dimension
    for (C d = 0; d < this->ndim_; ++d)
    {
      chooseParticleForDimension(index, d);
    }

    // If all chosen particles are 'index' particle,
    // then choose random particle for random dimension
    // exclude current 'index' particle
    if (allFiPartsEqualTo(index))
    {
      C dim = Generator<T, C>::getInt(0, this->ndim_ - 1);
      while (fi_[index][dim] == index)
      {
        fi_[index][dim] = Generator<T, C>::getInt(0, this->npar_ - 1);
      }
    }

    // Zero particle's time interval for update
    no_upd_times_[index] = 0;
  }
}

template <typename T, typename C>
T CLPSO_Swarm<T, C>::getRefreshingGap() const
{
  return refreshing_gap_;
}

template <typename T, typename C>
bool CLPSO_Swarm<T, C>::betweenBorders(const C & index) const
{
  C dim = this->ndim_;
  return std::all_of(&this->prts_->x[index * dim], &this->prts_->x[index * dim + dim],
    [&] (T value) {
      return this->b_lo_ < value && value < this->b_up_;
    }
  );
}

template <typename T, typename C>
bool CLPSO_Swarm<T, C>::allFiPartsEqualTo(const C & index) const
{
  return std::all_of(fi_[index].cbegin(), fi_[index].cend(), [&] (C value) {
    return value == index;
  });
}

template <typename T, typename C>
void CLPSO_Swarm<T, C>::chooseParticleForDimension(const C & index, const C & dim)
{
  if (Generator<T, C>::getReal(0., 1.) < pc_[index])
  {
    C p1;
    C p2;

    do
    {
      p1 = Generator<T, C>::getInt(0, this->npar_ - 1);
      p2 = Generator<T, C>::getInt(0, this->npar_ - 1);
    }
    while (p1 == p2 || p1 == index || p2 == index);

    fi_[index][dim] = better(this->prts_->f_best[p1], this->prts_->f_best[p2])
                      ? p1
                      : p2;
  }
  else
  {
    fi_[index][dim] = index;
  }
}
