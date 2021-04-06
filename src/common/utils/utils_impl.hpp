#include "../generator/generator.hpp"
#include "../../common/vector_types.hpp"
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <iostream>

template <typename T>
bool better(const T & value1, const T & value2)
{
  return value1 < value2;
}

template <typename T, typename C>
void shuffleIndexes(std::vector<C> & indexes)
{
  std::shuffle(indexes.begin(), indexes.end(), Generator<T, C>::getGenerator());
}

template <typename T, typename C>
std::vector<C> generateUniqueIntValuesExceptCurrent(const C & size,
                                                    const C & start,
                                                    const C & finish,
                                                    const C & excluded)
{
  std::vector<C> idxs;

  do
  {
    T value = Generator<T, C>::getInt(start, finish);

    if (value != excluded
      && std::find(idxs.begin(), idxs.end(), value) == idxs.end())
    {
      idxs.emplace_back(value);
    }
  }
  while (idxs.size() != size);

  return idxs;
}

template <typename T, typename C>
void printDataset(const dataset_type<T, C> & dataset)
{
  if (dataset.empty())
  {
    throw std::invalid_argument("Dataset is empty");
  }

  for (std::size_t i = 0; i < dataset.size(); ++i)
  {
    for (std::size_t j = 0; j < dataset[i].first.size(); ++j)
    {
      std::cout << dataset[i].first[j] << ' ';
    }

    for (std::size_t j = 0; j < dataset[i].second.size(); ++j)
    {
      std::cout << dataset[i].second[j];

      if (j != dataset[i].second.size() - 1)
      {
        std::cout << ' ';
      }
      else
      {
        std::cout << std::endl;
      }
    }
  }
}
