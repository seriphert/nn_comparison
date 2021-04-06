#ifndef NN_COMPARISON_VECTOR_TYPES
#define NN_COMPARISON_VECTOR_TYPES

#include <vector>

template <typename T>
using Vector1D = std::vector<T>;

template <typename T>
using Vector2D = std::vector<std::vector<T>>;

template <typename T>
using Vector3D = std::vector<std::vector<std::vector<T>>>;

#endif //NN_COMPARISON_VECTOR_TYPES
