#ifndef NN_COMPARISON_DATASET_HPP
#define NN_COMPARISON_DATASET_HPP

#include "dataset_type.hpp"
#include "../vector_types.hpp"
#include <cstddef>
#include <vector>
#include <string>
#include <utility>

template <typename T, typename C = std::size_t>
class Dataset
{
public:
  Dataset();
  void readDataset(const std::string & filename, const C & num_inputs,
                   const C & num_outputs);
  void normalizeInputs();
  void splitDataset(dataset_type<T, C> & train, dataset_type<T, C> & test,
                    const C & train_percentage);
  dataset_type<T, C> getDataset() const;

private:
  Vector2D<std::string> readDatasetValues(const std::string & filename);
  Vector1D<std::pair<T, T>> getMinmaxFromEachColumn();

  dataset_type<T, C> dataset_;
};

template class Dataset<float>;
template class Dataset<double>;

#endif //NN_COMPARISON_DATASET_HPP
