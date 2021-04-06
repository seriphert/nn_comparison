#ifndef NN_COMPARISON_PSO_LAYER_HPP
#define NN_COMPARISON_PSO_LAYER_HPP

#include "../base/layer.hpp"
#include "../../common/vector_types.hpp"
#include "../../common/activation_functions/activation_function.hpp"
#include <cstddef>
#include <memory>

template <typename T, typename C = std::size_t>
class PSO_Layer : public Layer<T>
{
public:
  PSO_Layer(const C & num_inputs, const C & num_ouputs,
            const std::shared_ptr<ActivationFunction<T>> & activ_func,
            const bool & is_bias_needed);
  PSO_Layer(const PSO_Layer & other);
  PSO_Layer(PSO_Layer && other) = delete;
  PSO_Layer& operator=(const PSO_Layer & other);
  PSO_Layer& operator=(PSO_Layer && other) = delete;
  ~PSO_Layer() override;
  Vector1D<T> predict(const Vector1D<T> & inputs,
                      typename Vector1D<T>::const_iterator startWeight) const;
};

template class PSO_Layer<float>;
template class PSO_Layer<double>;

# endif // NN_COMPARISON_PSO_LAYER_HPP
