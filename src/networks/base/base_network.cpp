#include "base_network.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>

template <typename T, typename C>
BaseNetwork<T, C>::BaseNetwork(const C & num_epochs,
                               const std::shared_ptr<LossFunction<T, C>> & loss_function,
                               const bool & is_biases_needed):
  add_bias_((is_biases_needed) ? 1: 0),
  is_builded_(false),
  num_inp_(0),
  num_out_(0),
  num_epochs_(num_epochs),
  layers_(0),
  loss_function_(loss_function)
{
}

template <typename T, typename C>
BaseNetwork<T, C>::BaseNetwork(const BaseNetwork<T, C> & other):
  add_bias_(other.add_bias_),
  is_builded_(other.is_builded_),
  num_inp_(other.num_inp_),
  num_out_(other.num_out_),
  num_epochs_(other.num_epochs_),
  layers_(other.layers_),
  loss_function_(other.loss_function_)
{
}

template <typename T, typename C>
BaseNetwork<T, C>& BaseNetwork<T, C>::operator=(const BaseNetwork<T, C> & other)
{
  if (this != &other)
  {
    this->add_bias_ = other.add_bias_;
    this->is_builded_ = other.is_builded_;
    this->num_inp_ = other.num_inp_;
    this->num_out_ = other.num_out_;
    this->num_epochs_ = other.num_epochs_;
    this->layers_ = other.layers_;
    this->loss_function_ = other.loss_function_;
  }
  return *this;
}

template <typename T, typename C>
BaseNetwork<T, C>::~BaseNetwork()
{
}

template <typename T, typename C>
T BaseNetwork<T, C>::accuracy(const dataset_type<T, C> & dataset) const
{
  T result = 0;

  if (dataset.empty())
  {
    return result;
  }

  for (C i = 0; i < dataset.size(); ++i)
  {
    Vector1D<T> inputs = dataset[i].first;
    Vector1D<C> exp_outputs = dataset[i].second;
    Vector1D<T> outputs = predict(inputs);

    // Find max element near to 1
    C max_idx = 0;
    for (C k = 0; k < outputs.size(); ++k)
    {
      max_idx = (outputs[max_idx] < outputs[k]) ? k : max_idx;
    }

    // Check if element in exp_output have 1 on max_idx
    result = (exp_outputs[max_idx] == 1) ? result + 1 : result;

    inputs.clear();
    outputs.clear();
    exp_outputs.clear();
  }

  return result / dataset.size();
}

template <typename T, typename C>
void BaseNetwork<T, C>::addLayer(const std::shared_ptr<Layer<T, C>> & layer)
{
  if (layer == nullptr)
  {
    throw std::invalid_argument("[addLayer] Nullptr as an input layer");
  }
  else if (layers_.size() != 0
        && layers_.back()->getNumOutputs() != layer->getNumInputs())
  {
    std::string err_msg = "[addLayer] Previous layer number of outputs" \
      " doesn't match with new layer's number of inputs ["
        + std::to_string(layers_.back()->getNumOutputs()) + " != "
          + std::to_string(layer->getNumInputs()) + "]";
    throw std::invalid_argument(err_msg);
  }

  layers_.emplace_back(layer);

  is_builded_ = false;
}

template <typename T, typename C>
Vector1D<T> BaseNetwork<T, C>::predict(const Vector1D<T> & inputs) const
{
  Vector1D<T> outputs = inputs;

  std::for_each(layers_.begin(), layers_.end(), [&] (const std::shared_ptr<Layer<T, C>> & layer) {
    outputs = layer->predict(outputs);
  });

  return outputs;
}

template <typename T, typename C>
Vector2D<T> BaseNetwork<T, C>::predict(const Vector2D<T> & inputs_vec) const
{
  Vector2D<T> outputs_vec;

  std::for_each(inputs_vec.begin(), inputs_vec.end(), [&] (const Vector1D<T> & inputs) {
    outputs_vec.emplace_back(predict(inputs));
  });

  return outputs_vec;
}

template <typename T, typename C>
Vector3D<T> BaseNetwork<T, C>::getWeights() const
{
  Vector3D<T> weights;

  std::for_each(layers_.begin(), layers_.end(), [&] (const std::shared_ptr<Layer<T, C>> & layer) {
    weights.emplace_back(layer->getWeights());
  });

  return weights;
}

template <typename T, typename C>
C BaseNetwork<T, C>::getNumEpochs() const
{
  return num_epochs_;
}

template <typename T, typename C>
C BaseNetwork<T, C>::getNumLayers() const
{
  return layers_.empty() ? 0 : layers_.size();
}

template <typename T, typename C>
bool BaseNetwork<T, C>::isBuilded() const
{
  return is_builded_;
}
