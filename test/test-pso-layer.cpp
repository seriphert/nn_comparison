#include "../src/layers/pso/pso_layer.hpp"
#include "../src/common/activation_functions/sigmoid.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>

template <typename T>
class PSO_Layer_Fixture : public ::testing::Test
{
protected:
  PSO_Layer_Fixture():
    enableBias_(true),
    num_inputs_(4),
    num_outputs_(5),
    weights_size_(num_inputs_ + ((enableBias_) ? 1 : 0)),
    layer_(num_inputs_, num_outputs_, std::make_shared<Sigmoid<T>>(), enableBias_)
  {
  }

  ~PSO_Layer_Fixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  bool enableBias_;
  std::size_t num_inputs_;
  std::size_t num_outputs_;
  std::size_t weights_size_;
  PSO_Layer<T, std::size_t> layer_;
};

TYPED_TEST_SUITE_P(PSO_Layer_Fixture);

TYPED_TEST_P(PSO_Layer_Fixture, initializationTest)
{
  // Check that variables defined for layer in fixture are
  // similar with initial fixture's variables
  EXPECT_EQ(this->num_inputs_, this->layer_.getNumInputs());
  EXPECT_EQ(this->num_outputs_, this->layer_.getNumOutputs());
  EXPECT_EQ(this->weights_size_, this->layer_.getWeights().size());
  
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(this->num_outputs_, this->layer_.getWeightsRow(i).size());
  }
}

TYPED_TEST_P(PSO_Layer_Fixture, copyConstructorTest)
{
  // Copy fixture's layer to new layer
  PSO_Layer<TypeParam> layer(this->layer_);

  // Check that copied layer variables equal to fixture layer
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights(), layer.getWeights());
}

TYPED_TEST_P(PSO_Layer_Fixture, copyOperatorTest)
{
  // Define new layer
  PSO_Layer<TypeParam> layer = {2, 2, std::make_shared<Sigmoid<TypeParam>>(), false};

  // Use copy operator to clone fixture's layer
  // in created layer
  layer = this->layer_;

  // Check that copied layer variables equal to fixture's layer variables
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights(), layer.getWeights());
}

TYPED_TEST_P(PSO_Layer_Fixture, copyOperatorSelfTest)
{
  // Define fixture's variables
  std::size_t num_inputs = this->num_inputs_;
  std::size_t num_outputs = this->num_outputs_;
  Vector2D<TypeParam> weights = this->layer_.getWeights();

  // Try to copy layer with itself
  this->layer_ = this->layer_;

  // Check that layer's variables are same
  EXPECT_EQ(num_inputs, this->layer_.getNumInputs());
  EXPECT_EQ(num_outputs, this->layer_.getNumOutputs());
  EXPECT_EQ(weights, this->layer_.getWeights());
}

TYPED_TEST_P(PSO_Layer_Fixture, predictTest)
{
  // Set initial weights
  std::size_t weights_vec_size = this->weights_size_ * this->num_outputs_;
  Vector1D<TypeParam> weights(weights_vec_size, 2);

  // Set layer inputs and expected outputs
  Vector1D<TypeParam> inputs(this->num_inputs_, 2);
  Vector1D<TypeParam> expected(this->num_outputs_, 1);

  // Calculate actual outputs values
  Vector1D<TypeParam> actual = this->layer_.predict(inputs, weights.cbegin());
  TypeParam tolerance = 1.e-6;

  // Check that actual varibales equal to expected
  // with some tolerance
  for (std::size_t i = 0; i < this->num_outputs_; ++i)
  {
    EXPECT_NEAR(expected[i], actual[i], tolerance);
  }
}

REGISTER_TYPED_TEST_SUITE_P
(
  PSO_Layer_Fixture,
  initializationTest,
  copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  predictTest
);

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(PSO_Layer_Suite, PSO_Layer_Fixture, test_types);
