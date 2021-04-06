#ifndef NN_COMPARISON_GENERATOR_HPP
#define NN_COMPARISON_GENERATOR_HPP

#include <cstddef>
#include <random>

template <typename T, typename C = std::size_t>
class Generator
{
public:
  static T getReal(const T & start, const T & end);
  static C getInt(const C & start, const C & end);
  static std::default_random_engine getGenerator();

private:
  static std::default_random_engine gen_;
};

template class Generator<float>;
template class Generator<double>;

#endif //NN_COMPARISON_GENERATOR_HPP
