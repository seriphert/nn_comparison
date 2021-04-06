#ifndef NN_COMPARISON_DATASET_TYPE_HPP
#define NN_COMPARISON_DATASET_TYPE_HPP

#include "../vector_types.hpp"
#include <cstddef>
#include <utility>

template <typename T, typename C = std::size_t>
using dataset_type = Vector1D<std::pair<Vector1D<T>, Vector1D<C>>>;

#endif //NN_COMPARISON_DATASET_TYPE_HPP
