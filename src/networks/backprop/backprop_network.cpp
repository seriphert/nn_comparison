#include "backprop_network.hpp"
#include "../../layers/backprop/backprop_layer.hpp"
#include "../../common/generator/generator.hpp"
#include <stdexcept>

template <typename T, typename C>
BackPropNetwork<T, C>::BackPropNetwork(const backprop_params_t<T, C> & params,
                                       const std::shared_ptr<LossFunction<T, C>> & loss_function,
                                       const bool & is_biases_needed):
  BaseNetwork<T, C>(params.num_epochs, loss_function, is_biases_needed),
  b_lo_(params.b_lo),
  b_up_(params.b_up),
  learn_rate_(params.learn_rate),
  momentum_(params.momentum)
{
}

template <typename T, typename C>
BackPropNetwork<T, C>::BackPropNetwork(const BackPropNetwork<T, C> & other):
  BaseNetwork<T, C>(other),
  b_lo_(other.b_lo_),
  b_up_(other.b_up_),
  learn_rate_(other.learn_rate_),
  momentum_(other.momentum_)
{
}

template <typename T, typename C>
BackPropNetwork<T, C>& BackPropNetwork<T, C>::operator=(const BackPropNetwork<T, C> & other)
{
  if (this != &other)
  {
    BaseNetwork<T, C>::operator=(other);
    this->b_lo_ = other.b_lo_;
    this->b_up_ = other.b_up_;
    this->learn_rate_ = other.learn_rate_;
    this->momentum_ = other.momentum_;
  }
  return *this;
}

template <typename T, typename C>
BackPropNetwork<T, C>::~BackPropNetwork() = default;

template <typename T, typename C>
void BackPropNetwork<T, C>::buildNetwork()
{
  if (this->layers_.empty())
  {
    throw std::logic_error("No layers added to model");
  }

  this->num_inp_ = this->layers_.front()->getNumInputs();
  this->num_out_ = this->layers_.back()->getNumOutputs();
  this->is_builded_ = true;
}

template <typename T, typename C>
void BackPropNetwork<T, C>::train(const dataset_type<T, C> & dataset)
{
  if (dataset.empty())
  {
    throw std::logic_error("Empty dataset");
  }
  else if (!this->is_builded_)
  {
    throw std::logic_error("Network didn't build yet");
  }

  // Generate weights for network layers and biases
  generateWeights();

  for (std::size_t epoch = 0; epoch < this->num_epochs_; ++epoch)
  {
    for (std::size_t i = 0; i < dataset.size(); ++i)
    {
      Vector1D<T> input = dataset[i].first;
      Vector1D<std::size_t> exp_output = dataset[i].second;

      Vector1D<T> output = predict(input);
      backpropagation(input, output, exp_output);

      input.clear();
      output.clear();
      exp_output.clear();
    }
  }
}

template <typename T, typename C>
T BackPropNetwork<T, C>::getBlow() const
{
  return b_lo_;
}

template <typename T, typename C>
T BackPropNetwork<T, C>::getBup() const
{
  return b_up_;
}

template <typename T, typename C>
T BackPropNetwork<T, C>::getLearnRate() const
{
  return learn_rate_;
}

template <typename T, typename C>
T BackPropNetwork<T, C>::getMomentum() const
{
  return momentum_;
}

template <typename T, typename C>
void BackPropNetwork<T, C>::generateWeights()
{
  for (std::size_t i = 0; i < this->layers_.size(); ++i)
  {
    std::size_t num_inputs = this->layers_[i]->getNumInputs() + this->add_bias_;
    std::size_t num_outputs = this->layers_[i]->getNumOutputs();

    for (std::size_t k = 0; k < num_inputs * num_outputs; ++k)
    {
      this->layers_[i]->getWeightsRow(k / num_outputs)[k % num_outputs]
        = Generator<T, C>::getReal(b_lo_, b_up_);
    }
  }
}

template <typename T, typename C>
Vector1D<T> BackPropNetwork<T, C>::predict(const Vector1D<T> & inputs)
{
  // Set current inputs
  Vector1D<T> inputs_vec = inputs;

  for (std::size_t i = 0; i < this->layers_.size(); ++i)
  {
    std::shared_ptr<BackPropLayer<T, C>> layer
      = std::dynamic_pointer_cast<BackPropLayer<T, C>>(this->layers_[i]);
    
    // Set current layer inputs
    layer->setInputs(inputs_vec);
    
    // Calculate inputs for next layer
    inputs_vec = layer->predict(inputs_vec);
  }

  // Return final vector of network outputs
  return inputs_vec;
}

template <typename T, typename C>
void BackPropNetwork<T, C>::backpropagation(const Vector1D<T> & inputs, const Vector1D<T> & outputs,
                                            const Vector1D<std::size_t> & exp_outputs)
{
  // Find deltas for outputs using last layer
  std::shared_ptr<BackPropLayer<T, C>> layer
    = std::dynamic_pointer_cast<BackPropLayer<T, C>>(this->layers_.back());

  // Calculate deltas for outputs
  Vector1D<T> delta_o = outputs;
  layer->applyDerivativeToValues(delta_o);

  for (std::size_t i = 0; i < this->num_out_; ++i)
  {
    delta_o[i] *= (static_cast<T>(exp_outputs[i]) - outputs[i]);
  }

  Vector1D<T> deltas = delta_o;

  // Calculate deltas for hidden layers (i != 0)
  for (std::size_t i = this->layers_.size() - 1; i > 0; --i)
  {
    layer = std::dynamic_pointer_cast<BackPropLayer<T, C>>(this->layers_[i]);
    
    // Calculate deltas for current layer
    layer->calculateDeltas(deltas);

    // Get deltas for using with next layer
    deltas = layer->getDeltas();
  }

  // Recalculate weights for each layer due to calculated earlier deltas  
  for (std::size_t i = 0; i < this->layers_.size(); ++i)
  {
    layer = std::dynamic_pointer_cast<BackPropLayer<T, C>>(this->layers_[i]);

    // If is not last layer, get deltas from next hidden layer,
    // else use output deltas
    deltas = (i != this->layers_.size() - 1)
            ? std::dynamic_pointer_cast<BackPropLayer<T, C>>(this->layers_[i + 1])->getDeltas()
            : delta_o;

    layer->recalculateWeights(learn_rate_, momentum_, deltas);
  }
}
