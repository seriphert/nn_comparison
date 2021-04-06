#ifndef NN_COMPARISON_UTILS_HPP
#define NN_COMPARISON_UTILS_HPP

#include "../dataset/dataset_type.hpp"
#include "../../common/vector_types.hpp"

template <typename T>
bool better(const T & value1, const T & value2);

template <typename T, typename C = std::size_t>
void shuffleIndexes(std::vector<C> & indexes);

template <typename T, typename C = std::size_t>
std::vector<C> generateUniqueIntValuesExceptCurrent(const C & size,
                                                    const C & start,
                                                    const C & finish,
                                                    const C & excluded);

template <typename T, typename C = std::size_t>
void printDataset(const dataset_type<T, C> & data);

#include "utils_impl.hpp"

#endif // NN_COMPARISON_UTILS_HPP
