#include "mean_squared_error.hpp"
#include <cmath>

template <typename T, typename C>
MeanSquaredError<T, C>::~MeanSquaredError() = default;

template <typename T, typename C>
T MeanSquaredError<T, C>::calculate(const Vector2D<C> & expected,
                                    const Vector2D<T> & actual)
{
  T result = 0;

  for (C i = 0; i < expected.size(); ++i)
  {
    for (C j = 0; j < expected[i].size(); ++j)
    {
      result += std::pow(static_cast<T>(expected[i][j]) - actual[i][j], 2.);
    }
  }

  return result / expected.size();
}
