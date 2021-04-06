#include "pso_network.hpp"
#include "../../common/vector_types.hpp"
#include "../../swarms/pso/pso_swarm.hpp"
#include <stdexcept>

template <typename T, typename C>
PSO_Network<T, C>::PSO_Network(const pso_params_t<T, C> & params,
                               const std::shared_ptr<LossFunction<T, C>> & loss_function,
                               const bool & is_biases_needed):
  BasePSO_Network<T, C>(params.num_iters, params.weight_func, loss_function, is_biases_needed)
{
  this->swarm_ = std::make_shared<PSO_Swarm<T, C>>(params);
}

template <typename T, typename C>
PSO_Network<T, C>::PSO_Network(const PSO_Network<T, C> & other):
  BasePSO_Network<T, C>(other)
{
}

template <typename T, typename C>
PSO_Network<T, C>& PSO_Network<T, C>::operator=(const PSO_Network<T, C> & other)
{
  if (this != &other)
  {
    BasePSO_Network<T, C>::operator=(other);
  }
  return *this;
}

template <typename T, typename C>
PSO_Network<T, C>::~PSO_Network() = default;

template <typename T, typename C>
void PSO_Network<T, C>::train(const dataset_type<T, C> & dataset)
{
  if (dataset.empty())
  {
    throw std::logic_error("Empty dataset");
  }
  else if (!this->is_builded_)
  {
    throw std::logic_error("Network didn't build yet");
  }

  // Set dataset to work with for Mean Squared error
  this->dataset_ = dataset;

  // Form inputs and expected outputs vectors
  Vector2D<T> inputs_vec;
  Vector2D<std::size_t> exp_outputs_vec;

  for (std::size_t i = 0; i < this->dataset_.size(); ++i)
  {
    inputs_vec.emplace_back(this->dataset_[i].first);
    exp_outputs_vec.emplace_back(this->dataset_[i].second);
  }

  this->swarm_->initializeSwarm();

  // Working process
  for (std::size_t epoch = 0; epoch < this->num_epochs_; ++epoch)
  {
    // Recalculate weight for current iteration
    T weight = (*this->weight_func_)(epoch, this->num_epochs_);

    // Sequential update for each particle
    for (std::size_t i = 0; i < this->swarm_->getNumParticles(); ++i)
    {
      typename Vector1D<T>::const_iterator iter = this->swarm_->getStartWeightsIterator(i);
      // Get outputs vector for loss calculation
      Vector2D<T> outputs_vec = this->predict(inputs_vec, iter);
      // Calculate new function value for current particles 'x' values
      T func_value = this->loss_function_->calculate(exp_outputs_vec, outputs_vec);

      if (this->swarm_->tryToUpdateBestParticleState(i, func_value))
      {
        this->swarm_->tryToUpdateBestParticle(i);
      }

      // Recalculate new particle's parameters (velocity and placement)
      this->swarm_->updateParticle(i, weight);
    }
  }

  // Set best particles's state as weights of the network
  std::size_t best_idx = this->swarm_->getBestIndex();
  typename Vector1D<T>::const_iterator iter = this->swarm_->getStartWeightsIterator(best_idx);
  this->setWeights(iter);

  // Clear unused now memory
  this->swarm_->clearUnusedMemory();
  this->dataset_.clear();
}

template <typename T, typename C>
T PSO_Network<T, C>::getSocWeight() const
{
  return std::dynamic_pointer_cast<PSO_Swarm<T, C>>(this->swarm_)->getSocWeight();
}
