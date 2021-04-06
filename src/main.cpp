#include "common/dataset/dataset.hpp"
#include "common/params_types.hpp"
#include "common/networks.hpp"
#include "common/layers.hpp"
#include "common/enum/network_type.hpp"
#include "common/enum/test_dataset.hpp"
#include "common/activation_functions.hpp"
#include "common/loss_functions.hpp"
#include "common/inertia_weights.hpp"
#include "common/vector_types.hpp"
#include "common/utils/utils.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <ctime>
#include <numeric>
#include <string>
#include <iostream>

std::string datasets_directory = "../datasets/";

template <typename T>
void initializeNetworkParameters(T & num_inputs, std::vector<T> & hiddens,
                                 T & num_outputs,
                                 std::string & filename,
                                 const TestDataset & test_dataset)
{
  switch (test_dataset)
  {
    case IRIS:
    {
      filename = datasets_directory + "iris.csv";
      num_inputs  = 4;
      hiddens = {7, 5, 4};
      num_outputs = 3;
      break;
    }

    case WHEAT_SEEDS:
    {
      filename = datasets_directory + "wheat-seeds.csv";
      num_inputs  = 7;
      hiddens = {8, 6, 4};
      num_outputs = 3;
      break;
    }

    case THYROID:
    {
      filename = datasets_directory + "new-thyroid.csv";
      num_inputs  = 5;
      hiddens = {7, 5, 4};
      num_outputs = 3;
      break;
    }

    case WINE:
    {
      filename = datasets_directory + "wine.csv";
      num_inputs  = 13;
      hiddens = {6, 4, 3};
      num_outputs = 3;
      break;
    }

    case CICADA:
    {
      filename = datasets_directory + "cicada.csv";
      num_inputs  = 5;
      hiddens = {7, 5, 3};
      num_outputs = 3;
      break;
    }
  }
}

template <typename T>
void initializeSGDParameters(backprop_params_t<T> & params,
                             const T & b_lo,
                             const T & b_up,
                             const std::size_t & num_epochs,
                             const TestDataset & test_dataset)
{
  params.b_lo = b_lo;
  params.b_up = b_up;
  params.num_epochs = num_epochs;
  
  switch (test_dataset)
  {
    case IRIS:
    {
      params.learn_rate = 0.5;
      params.momentum   = 0.9;
      break;
    }

    case WHEAT_SEEDS:
    {
      params.learn_rate = 0.5;
      params.momentum   = 0.9;
      break;
    }

    case THYROID:
    {
      params.learn_rate = 0.5;
      params.momentum   = 0.9;
      break;
    }

    case WINE:
    {
      params.learn_rate = 0.3;
      params.momentum   = 0.9;
      break;
    }

    case CICADA:
    {
      params.learn_rate = 0.3;
      params.momentum   = 0.01;
      break;
    }
  }
}

template <typename T>
void readAndSplitDatasetToTrainAndTest(const std::string & filename,
                                       const std::size_t & num_inputs,
                                       const std::size_t & num_outputs,
                                       dataset_type<T> & train,
                                       dataset_type<T> & test,
                                       const std::size_t & train_percentage)
{
  Dataset<T> dataset;
  dataset.readDataset(filename, num_inputs, num_outputs);
  //printDataset(dataset.getDataset());

  // Normalize input values of dataset
  dataset.normalizeInputs();

  // Split dataset into train and test datasets
  dataset.splitDataset(train, test, train_percentage);
}

template <typename T>
void addLayersToBackPropNetwork(std::shared_ptr<BaseNetwork<T>> & network,
                                const std::size_t & num_inputs,
                                const Vector1D<std::size_t> & hiddens,
                                const std::size_t & num_outputs,
                                const std::shared_ptr<ActivationFunction<T>> & activ_func,
                                const bool & enable_biases)
{
  std::size_t layes_cnt = hiddens.size() + 1;
  
  if (layes_cnt != 1)
  {
    for (std::size_t j = 0; j < layes_cnt; ++j)
    {
      if (j > 0 && j < hiddens.size())
      {
        network->addLayer(std::make_shared<BackPropLayer<T>>(hiddens[j - 1], hiddens[j],
                          activ_func, enable_biases));
      }
      else if (j == 0)
      {
        network->addLayer(std::make_shared<BackPropLayer<T>>(num_inputs, hiddens[j], 
                          activ_func, enable_biases));
      }
      else
      {
        network->addLayer(std::make_shared<BackPropLayer<T>>(hiddens[j - 1], num_outputs,
                          activ_func, enable_biases));
      }
    }
  }
  else
  {
    network->addLayer(std::make_shared<BackPropLayer<T>>(num_inputs, num_outputs,
                      activ_func, enable_biases));
  }
}

template <typename T>
void addLayersToPSO_Network(std::shared_ptr<BaseNetwork<T>> & network,
                            const std::size_t & num_inputs,
                            const Vector1D<std::size_t> & hiddens,
                            const std::size_t & num_outputs,
                            const std::shared_ptr<ActivationFunction<T>> & activ_func,
                            const bool & enable_biases)
{
  std::size_t layes_cnt = hiddens.size() + 1;
  
  if (layes_cnt != 1)
  {
    for (std::size_t j = 0; j < layes_cnt; ++j)
    {
      if (j > 0 && j < hiddens.size())
      {
        network->addLayer(std::make_shared<PSO_Layer<T>>(hiddens[j - 1], hiddens[j],
                          activ_func, enable_biases));
      }
      else if (j == 0)
      {
        network->addLayer(std::make_shared<PSO_Layer<T>>(num_inputs, hiddens[j], 
                          activ_func, enable_biases));
      }
      else
      {
        network->addLayer(std::make_shared<PSO_Layer<T>>(hiddens[j - 1], num_outputs,
                          activ_func, enable_biases));
      }
    }
  }
  else
  {
    network->addLayer(std::make_shared<PSO_Layer<T>>(num_inputs, num_outputs,
                      activ_func, enable_biases));
  }
}

template <typename T>
T standartDeviation(const Vector1D<T> & values, const T & mean)
{
  T sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
  return std::sqrt(sum / values.size() - mean * mean);
}

template <typename T>
void startNetworkTraning(const NetworkType & network_type, const TestDataset & test_dataset,
                         const std::size_t & program_iters,
                         const std::size_t & num_epochs,
                         const bool & enable_biases)
{
  // Variable network parameters
  std::size_t num_inputs;
  Vector1D<std::size_t> hiddens;
  std::size_t num_outputs;
  std::string filename;
  // Initialize network parameters
  initializeNetworkParameters(num_inputs, hiddens, num_outputs, filename, test_dataset);

  // Parameters for networks
  backprop_params_t<T> backprop_params;
  pso_params_t<T> pso_params;
  clpso_params_t<T> clpso_params;

  // Tuning parametes for all networks
  T b_lo = -5;
  T b_up = -b_lo;

  // Tuning parametes only for PSO-type networks
  T cog_coeff = 1.49445;
  std::size_t num_particles = 10;
  std::size_t refreshing_gap = 7;  // for (I)CLPSO
  const T w0 = 0.9;
  const T w1 = 0.4;
  std::shared_ptr<InertiaWeight<T>> weight_func
      = std::make_shared<LinearDecreasingInertiaWeight<T>>(w0, w1);
  
  // Activation and loss functions
  std::shared_ptr<ActivationFunction<T>> activ_func = std::make_shared<Sigmoid<T>>();
  std::shared_ptr<LossFunction<T>> loss_function = std::make_shared<MeanSquaredError<T>>();

  // Initialize tuning parameters due to network type
  if (network_type == NetworkType::BACKPROPAGATION)
  {
    initializeSGDParameters(backprop_params, b_lo, b_up, num_epochs, test_dataset);
  }
  else if (network_type == NetworkType::PSO)
  { 
    pso_params.num_particles = num_particles;
    pso_params.b_lo = b_lo;
    pso_params.b_up = b_up;
    pso_params.cog_weight = cog_coeff;
    pso_params.soc_weight = cog_coeff;
    pso_params.num_iters = num_epochs;
    pso_params.weight_func = weight_func;
  }
  else
  {
    clpso_params.num_particles = num_particles;
    clpso_params.b_lo = b_lo;
    clpso_params.b_up = b_up;
    clpso_params.cog_weight = cog_coeff;
    clpso_params.refreshing_gap = refreshing_gap;
    clpso_params.num_iters = num_epochs;
    clpso_params.weight_func = weight_func;
  }

  // Initialize train and test datasets
  dataset_type<T> train;
  dataset_type<T> test;
  std::size_t train_percentage = 80;
  readAndSplitDatasetToTrainAndTest(filename, num_inputs, num_outputs,
                                    train, test, train_percentage);

  Vector1D<T> train_values;
  Vector1D<T> test_values;
  Vector1D<T> time_values;

  std::shared_ptr<BaseNetwork<T>> network;
  for (std::size_t i = 0; i < program_iters; ++i)
  {
    switch (network_type)
    {
      case BACKPROPAGATION:
      {
        network = std::make_shared<BackPropNetwork<T>>(backprop_params, loss_function, enable_biases);
        break;
      }

      case PSO:
      {
        network = std::make_shared<PSO_Network<T>>(pso_params, loss_function, enable_biases);
        break;
      }

      case CLPSO:
      {
        network = std::make_shared<CLPSO_Network<T>>(clpso_params, loss_function, enable_biases);
        break;
      }

      case ICLPSO:
      {
        network = std::make_shared<ICLPSO_Network<T>>(clpso_params, loss_function, enable_biases);
        break;
      }
    }

    if (network_type == BACKPROPAGATION)
    {
      addLayersToBackPropNetwork(network, num_inputs, hiddens, num_outputs,
                                 activ_func, enable_biases);
    }
    else
    {
      addLayersToPSO_Network(network, num_inputs, hiddens, num_outputs,
                            activ_func, enable_biases);
    }

    network->buildNetwork();

    std::cout << "Iteration " << i + 1 << " :: ";

    T start = static_cast<T>(clock()) / CLOCKS_PER_SEC;
    network->train(train);
    T finish = static_cast<T>(clock()) / CLOCKS_PER_SEC;

    train_values.emplace_back(network->accuracy(train));
    test_values.emplace_back(network->accuracy(test));
    time_values.emplace_back(finish - start);
    std::cout << " [Diff] : " << time_values.back() << std::endl;

    network.reset();
  }

  std::cout << std::endl;

  T mean = std::accumulate(train_values.begin(), train_values.end(), 0.0) / train_values.size();
  std::cout << "Mean train value : " << mean << std::endl;
  std::cout << "SD train value : " << standartDeviation(train_values, mean) << std::endl << std::endl;
  
  mean = std::accumulate(test_values.begin(), test_values.end(), 0.0) / test_values.size();
  std::cout << "Mean test value : " << mean << std::endl;
  std::cout << "SD test value : " << standartDeviation(test_values, mean) << std::endl << std::endl;

  mean = std::accumulate(time_values.begin(), time_values.end(), 0.0) / time_values.size();
  std::cout << "Average time : " << mean << std::endl;
  std::cout << "SD time : " << standartDeviation(time_values, mean) << std::endl << std::endl;
}

int main(int argc, char * argv[])
{
  // Program paramers
  std::size_t program_iters = 1;

  // Common network parametes
  NetworkType network_type = NetworkType::PSO;
  TestDataset test_dataset = TestDataset::IRIS;
  std::size_t num_epochs = 20;
  bool enable_biases = true;

  startNetworkTraning<double>(network_type, test_dataset, program_iters,
                       num_epochs, enable_biases);
  return 0;
}
