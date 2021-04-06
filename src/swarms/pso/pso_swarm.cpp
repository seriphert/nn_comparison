#include "pso_swarm.hpp"
#include "../../common/generator/generator.hpp"
#include "../../common/utils/utils.hpp"
#include <cmath>

template <typename T, typename C>
PSO_Swarm<T, C>::PSO_Swarm(const pso_params_t<T, C> & params):
  Swarm<T, C>(params),
  soc_weight_(params.soc_weight)
{
}

template <typename T, typename C>
PSO_Swarm<T, C>::PSO_Swarm(const PSO_Swarm<T, C> & other):
  Swarm<T, C>(other),
  soc_weight_(other.soc_weight_)
{
}

template <typename T, typename C>
PSO_Swarm<T, C> & PSO_Swarm<T, C>::operator=(const PSO_Swarm<T, C> & other)
{
  if (this != &other)
  {
    Swarm<T, C>::operator=(other);
    this->soc_weight_ = other.soc_weight_;
  }
  return *this;
}

template <typename T, typename C>
PSO_Swarm<T, C>::~PSO_Swarm() = default;

template <typename T, typename C>
void PSO_Swarm<T, C>::updateParticle(const C & index, const T & weight)
{
  C dim = this->ndim_;

  for (C d = 0; d < dim; ++d)
  {
    T r1 = Generator<T, C>::getReal(0., 1.);
    T r2 = Generator<T, C>::getReal(0., 1.);

    T temp = weight * this->prts_->v[index * dim + d]
            + this->cog_weight_ * r1 * (this->prts_->p[index * dim + d] - this->prts_->x[index * dim + d])
            + soc_weight_ * r2 * (this->prts_->p[this->best_idx_ * dim + d] - this->prts_->x[index * dim + d]);

    temp = (std::abs(temp) > this->v_max_)
          ? ((temp > this->v_max_) ? this->v_max_ : this->v_min_)
          : temp;

    this->prts_->x[index * dim + d] += (this->prts_->v[index * dim + d] = temp);
  }
}
 template <typename T, typename C>
bool PSO_Swarm<T, C>::tryToUpdateBestParticleState(const C & index, const T & func_value)
{
  C dim = this->ndim_;
  this->prts_->f_curr[index] = func_value;

  if (better(this->prts_->f_curr[index], this->prts_->f_best[index]))
  {
    this->prts_->f_best[index] = this->prts_->f_curr[index];
    std::copy(&this->prts_->x[index * dim], &this->prts_->x[index * dim + dim],
                &this->prts_->p[index * dim]);
    
    // Successful particle's best value update, return 'true'
    return true;
  }

  // Unsuccessful particle's best value update, return 'false'
  return false;
}

template <typename T, typename C>
void PSO_Swarm<T, C>::clearUnusedMemory()
{
  this->prts_.reset();
  this->prts_ = nullptr;
}

template <typename T, typename C>
T PSO_Swarm<T, C>::getSocWeight() const
{
  return soc_weight_;
}
