#include "sigmoid.hpp"
#include <algorithm>
#include <cmath>

template <typename T>
Sigmoid<T>::~Sigmoid() = default;

template <typename T>
void Sigmoid<T>::calculate(Vector1D<T> & args) const
{
  std::for_each(args.begin(), args.end(), [] (T & arg) { arg = 1 / (1 + std::exp(-arg)); });
}

template <typename T>
void Sigmoid<T>::derivative(Vector1D<T> & args) const
{
  std::for_each(args.begin(), args.end(), [] (T & arg) { arg = (1 - arg) * arg; });
}
