#include "dataset.hpp"
#include "../utils/utils.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

template <typename T, typename C>
Dataset<T, C>::Dataset():
  dataset_(0)
{
}

template <typename T, typename C>
void Dataset<T, C>::readDataset(const std::string & filename, const C & num_inputs,
                                const C & num_outputs)
{
  // Read all dataset include lables from file
  Vector2D<std::string> data = readDatasetValues(filename);

  // Values length in first dataset row
  C values_length = data[0].size();

  // Check if all rows have same length
  if (!std::all_of(data.cbegin(), data.cend(), [&] (std::vector<std::string> vec) {
    return vec.size() == values_length;
  }))
  {
    throw std::invalid_argument("[readDataset] Dataset rows lengths must be equal");
  }

  if (!dataset_.empty())
  {
    dataset_.clear();
  }

  dataset_.resize(data.size());

  // Add values to dataset
  for (C i = 0; i < data.size(); ++i)
  {
    for (C j = 0; j < num_inputs + num_outputs; ++j)
    {
      // Place data into dataset
      std::stringstream ss;
      ss.str(data[i][j]);

      if (j < num_inputs)
      {
        // Floating point input value
        T val;
        ss >> val;
        dataset_[i].first.emplace_back(val);
      }
      else
      {
        // Binary output value
        C val;
        ss >> val;
        dataset_[i].second.emplace_back(val);
      }

      ss.clear();
    }
  }
}

template <typename T, typename C>
void Dataset<T, C>::normalizeInputs()
{
  if (dataset_.empty())
  {
    throw std::logic_error("[normalizeInputs] Read dataset first");
  }

  // Get minimal and maximal values for each column in dataset
  Vector1D<std::pair<T, T>> minmax = getMinmaxFromEachColumn();

  // Normalize input value with found earlier minmax values
  for (C i = 0; i < dataset_[0].first.size(); ++i)
  {
    for (C k = 0; k < dataset_.size(); ++k)
    {
      dataset_[k].first[i] = (dataset_[k].first[i] - minmax[i].first)
                              / (minmax[i].second - minmax[i].first);
    }
  }
}

template <typename T, typename C>
void Dataset<T, C>::splitDataset(dataset_type<T, C> & train, dataset_type<T, C> & test,
                                 const C & train_percentage)
{
  if (dataset_.empty())
  {
    throw std::logic_error("[splitDataset] Read dataset first");
  }
  else if (!train.empty())
  {
    throw std::logic_error("[splitDataset] Train data is not empty");
  }
  else if (!test.empty())
  {
    throw std::logic_error("[splitDataset] Test data is not empty");
  }

  std::vector<C> indexes(dataset_.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  shuffleIndexes<T, C>(indexes);

  for (C i = 0; i < dataset_.size(); ++i)
  {
    if (i < dataset_.size() * static_cast<T>(train_percentage) / 100)
    {
      train.emplace_back(dataset_[indexes[i]]);
    }
    else
    {
      test.emplace_back(dataset_[indexes[i]]);
    }
  }
}

template <typename T, typename C>
dataset_type<T, C> Dataset<T, C>::getDataset() const
{
  return dataset_;
}

template <typename T, typename C>
Vector2D<std::string> Dataset<T, C>::readDatasetValues(const std::string & filename)
{
  std::ifstream file(filename);

  if (!file.is_open())
  {
    throw std::invalid_argument("[readDatasetValues] File [" + filename + "] doesn't exist");
  }

  std::string line;
  Vector2D<std::string> data;

  if (!file.eof() && std::getline(file, line))
  {
    // Skip first labels line
  }

  while (!file.eof() && std::getline(file, line))
  {
    std::string token;
    std::stringstream ss;
    Vector1D<std::string> tokens;

    ss.str(line);

    while (ss >> token)
    {
      tokens.emplace_back(token);
    }
    
    data.emplace_back(tokens);
    ss.clear();
    tokens.clear();
  }

  return data;
}

template <typename T, typename C>
Vector1D<std::pair<T, T>> Dataset<T, C>::getMinmaxFromEachColumn()
{
  Vector1D<std::pair<T, T>> minmax;

  for (C i = 0; i < dataset_[0].first.size(); ++i)
  {
    T min_value = dataset_[0].first[i];
    T max_value = dataset_[0].first[i];

    for (C k = 1; k < dataset_.size(); ++k)
    {
      if (min_value > dataset_[k].first[i])
      {
        min_value = dataset_[k].first[i];
      }
      else if (max_value < dataset_[k].first[i])
      {
        max_value = dataset_[k].first[i];
      }
    }

    minmax.emplace_back(std::make_pair(min_value, max_value));
  }

  return minmax;
}
