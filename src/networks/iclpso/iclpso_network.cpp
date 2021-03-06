#include "iclpso_network.hpp"
#include "../../common/vector_types.hpp"
#include "../../swarms/iclpso/iclpso_swarm.hpp"
#include <stdexcept>

template <typename T, typename C>
ICLPSO_Network<T, C>::ICLPSO_Network(const clpso_params_t<T, C> & params,
                                     const std::shared_ptr<LossFunction<T, C>> & loss_function,
                                     const bool & is_biases_needed):
  CLPSO_Network<T, C>(params, loss_function, is_biases_needed)
{
  this->swarm_ = std::make_shared<ICLPSO_Swarm<T, C>>(params);
}

template <typename T, typename C>
ICLPSO_Network<T, C>::ICLPSO_Network(const ICLPSO_Network<T, C> & other):
  CLPSO_Network<T, C>(other)
{
}

template <typename T, typename C>
ICLPSO_Network<T, C>& ICLPSO_Network<T, C>::operator=(const ICLPSO_Network<T, C> & other)
{
  if (this != &other)
  {
    CLPSO_Network<T, C>::operator=(other);
  }
  return *this;
}

template <typename T, typename C>
ICLPSO_Network<T, C>::~ICLPSO_Network() = default;

template <typename T, typename C>
void ICLPSO_Network<T, C>::train(const dataset_type<T, C> & dataset)
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

  // Initialize swarm components
  std::shared_ptr<ICLPSO_Swarm<T, C>> swarm
      = std::dynamic_pointer_cast<ICLPSO_Swarm<T, C>>(this->swarm_);
  swarm->initializeSwarm();
  swarm->initializeFiArray();
  swarm->initializeNoUpdateTimes();
  swarm->initializeDescendingIndexes();

  // Working process
  for (std::size_t epoch = 0; epoch < this->num_epochs_; ++epoch)
  {
    // Recalculate weight for current iteration
    T weight = (*this->weight_func_)(epoch, this->num_epochs_);

    if (swarm->anyParticleNeedsToBeUpdated())
    {
      swarm->generateProbabilities();
    }

    // Sequential update for each particle
    for (std::size_t i = 0; i < this->swarm_->getNumParticles(); ++i)
    {
      typename Vector1D<T>::const_iterator iter = swarm->getStartWeightsIterator(i);
      // Get outputs vector for loss calculation
      Vector2D<T> outputs_vec = this->predict(inputs_vec, iter);
      // Calculate new function value for current particles 'x' values
      T func_value = this->loss_function_->calculate(exp_outputs_vec, outputs_vec);

      // Update particle's best value and, if available,
      // update best particle index
      if (swarm->tryToUpdateBestParticleState(i, func_value))
      {
        swarm->tryToUpdateBestParticle(i);
      }

      // Generate size for array of random
      // selected particles to learn from
      swarm->generateNumberOfParticlesForLearn(i, epoch);

      // Choose new particles indexes for each
      // dimension to update current particle
      swarm->chooseParticlesToLearnFrom(i, epoch);

      // Recalculate new particle's parameters (velocity and placement)
      swarm->updateParticle(i, weight);
    }
  }

  // Set best particles's state as weights of the network
  std::size_t best_idx = swarm->getBestIndex();
  typename Vector1D<T>::const_iterator iter = swarm->getStartWeightsIterator(best_idx);
  this->setWeights(iter);

  // Clear unused now memory
  this->swarm_->clearUnusedMemory();
  this->dataset_.clear();
}
