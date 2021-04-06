#include "generator.hpp"
#include <chrono>

template <typename T, typename C>
std::default_random_engine Generator<T, C>::gen_ = std::default_random_engine(
  std::chrono::system_clock::now().time_since_epoch().count()
);

template <typename T, typename C>
T Generator<T, C>::getReal(const T & start, const T & end)
{
  std::uniform_real_distribution<T> dist(start, end);
  return dist(gen_);
}

template <typename T, typename C>
C Generator<T, C>::getInt(const C & start, const C & end)
{
  std::uniform_int_distribution<C> dist(start, end);
  return dist(gen_);
}

template <typename T, typename C>
std::default_random_engine Generator<T, C>::getGenerator()
{
  return gen_;
}
