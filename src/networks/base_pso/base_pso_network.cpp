#include "base_pso_network.hpp"
#include "../../common/vector_types.hpp"
#include "../../layers/pso/pso_layer.hpp"
#include <stdexcept>
#include <algorithm>

template <typename T, typename C>
BasePSO_Network<T, C>::BasePSO_Network(const C & num_iters,
                                       const std::shared_ptr<InertiaWeight<T, C>> & weight_func,
                                       const std::shared_ptr<LossFunction<T, C>> & loss_function,
                                       const bool & is_biases_needed):
  BaseNetwork<T, C>(num_iters, loss_function, is_biases_needed),
  dataset_(0),
  swarm_(nullptr),
  weight_func_(weight_func)
{
}

template <typename T, typename C>
BasePSO_Network<T, C>::BasePSO_Network(const BasePSO_Network<T, C> & other):
  BaseNetwork<T, C>(other),
  dataset_(other.dataset_),
  swarm_(other.swarm_),
  weight_func_(other.weight_func_)
{
}

template <typename T, typename C>
BasePSO_Network<T, C>& BasePSO_Network<T, C>::operator=(const BasePSO_Network<T, C> & other)
{
  if (this != &other)
  {
    BaseNetwork<T, C>::operator=(other);
    this->dataset_ = other.dataset_;
    this->swarm_ = other.swarm_;
    this->weight_func_ = other.weight_func_;
  }
  return *this;
}

template <typename T, typename C>
BasePSO_Network<T, C>::~BasePSO_Network()
{
}

template <typename T, typename C>
void BasePSO_Network<T, C>::buildNetwork()
{
  if (this->layers_.empty())
  {
    throw std::logic_error("No layers added to model");
  }

  this->num_inp_ = this->layers_.front()->getNumInputs();
  this->num_out_ = this->layers_.back()->getNumOutputs();

  // Calculate number of dimensions used for each particle
  C ndim = 0;
  std::for_each(this->layers_.begin(), this->layers_.end(), [&] (const std::shared_ptr<Layer<T, C>> & layer) {
    ndim += (layer->getNumInputs() + this->add_bias_) * layer->getNumOutputs();
  });

  // Set calculated dimensions to swarm
  swarm_->setNumDimensions(ndim);
  this->is_builded_ = true;
}

template <typename T, typename C>
C BasePSO_Network<T, C>::getNumParticles() const
{
  return swarm_->getNumParticles();
}

template <typename T, typename C>
C BasePSO_Network<T, C>::getNumDimensions() const
{
  return swarm_->getNumDimensions();
}

template <typename T, typename C>
T BasePSO_Network<T, C>::getCogWeight() const
{
  return swarm_->getCogWeight();
}

template <typename T, typename C>
T BasePSO_Network<T, C>::getBlo() const
{
  return swarm_->getBlo();
}

template <typename T, typename C>
T BasePSO_Network<T, C>::getBup() const
{
  return swarm_->getBup();
}

template <typename T, typename C>
void BasePSO_Network<T, C>::setWeights(typename Vector1D<T>::const_iterator startWeightsIterator)
{
  C idx = 0;

  for (std::shared_ptr<Layer<T, C>> & layer : this->layers_)
  {
    C num_inp = layer->getNumInputs() + this->add_bias_;
    C num_out = layer->getNumOutputs();

    for (C j = 0; j < num_inp * num_out; ++j)
    {
      layer->getWeightsRow(j / num_out)[j % num_out] = *(startWeightsIterator + idx + j);
    }

    idx += num_inp * num_out;
  }
}

template <typename T, typename C>
Vector1D<T> BasePSO_Network<T, C>::predict(const Vector1D<T> & inputs,
                                           typename Vector1D<T>::const_iterator weight_iter) const
{
  C idx = 0;
  Vector1D<T> outputs = inputs;

  for (C i = 0; i < this->layers_.size(); ++i)
  {
    std::shared_ptr<PSO_Layer<T, C>> layer
        = std::dynamic_pointer_cast<PSO_Layer<T, C>>(this->layers_[i]);

    outputs = layer->predict(outputs, weight_iter + idx);

    C num_inp = layer->getNumInputs() + this->add_bias_;
    C num_out = layer->getNumOutputs();
    idx += num_inp * num_out;
  }

  return outputs;
}

template <typename T, typename C>
Vector2D<T> BasePSO_Network<T, C>::predict(const Vector2D<T> & inputs_vec,
                                           typename Vector1D<T>::const_iterator weight_iter) const
{
  Vector2D<T> outputs_vec;

  std::for_each(inputs_vec.begin(), inputs_vec.end(), [&] (const Vector1D<T> & inputs) {
    outputs_vec.emplace_back(predict(inputs, weight_iter));
  });

  return outputs_vec;
}
