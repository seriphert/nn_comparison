#include "../src/layers/base/layer.hpp"
#include "../src/common/activation_functions/sigmoid.hpp"
#include <cstddef>
#include <memory>
#include <gtest/gtest.h>

template <typename T>
class LayerFixture : public ::testing::Test
{
protected:
  LayerFixture():
    enableBias_(true),
    num_inputs_(4),
    num_outputs_(5),
    weights_size_(num_inputs_ + ((enableBias_) ? 1 : 0)),
    layer_(num_inputs_, num_outputs_, std::make_shared<Sigmoid<T>>(), enableBias_)
  {
  }

  ~LayerFixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  bool enableBias_;
  std::size_t num_inputs_;
  std::size_t num_outputs_;
  std::size_t weights_size_;
  Layer<T, std::size_t> layer_;
};

TYPED_TEST_SUITE_P(LayerFixture);

TYPED_TEST_P(LayerFixture, initializationTest)
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

TYPED_TEST_P(LayerFixture, copyConstructorTest)
{
  // Copy fixture's layer to new layer
  Layer<TypeParam> layer(this->layer_);

  // Check that copied layer variables equal to fixture layer
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights(), layer.getWeights());
}

TYPED_TEST_P(LayerFixture, copyOperatorTest)
{
  // Define new layer
  Layer<TypeParam> layer = {2, 2, std::make_shared<Sigmoid<TypeParam>>(), false};
  
  // Use copy operator to clone fixture's layer
  // in created layer
  layer = this->layer_;

  // Check that copied layer variables equal to fixture's layer variables
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights(), layer.getWeights());
}

TYPED_TEST_P(LayerFixture, copyOperatorSelfTest)
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

TYPED_TEST_P(LayerFixture, getWeightsRowTest)
{
  // Set weights vector
  Vector1D<TypeParam> weights(this->weights_size_, 1);

  // Init layer weights with weights vector
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    this->layer_.getWeightsRow(i) = weights;
  }

  // Check that all layer weights rows equal to initial vector
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(weights, this->layer_.getWeightsRow(i));
  }
}

// Similar to getWeightsRow, but get all weights
TYPED_TEST_P(LayerFixture, getWeightsTest)
{
  Vector1D<TypeParam> weights(this->weights_size_, 1);

  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    this->layer_.getWeightsRow(i) = weights;
  }

  Vector2D<TypeParam> layer_weights = this->layer_.getWeights();

  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(weights, layer_weights[i]);
  }
}

TYPED_TEST_P(LayerFixture, predictTest)
{
  // Set initial weights
  Vector1D<TypeParam> weights(this->weights_size_, 2);

  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    this->layer_.getWeightsRow(i) = weights;
  }

  // Set parameters : inputs, outputs, expected outputs
  Vector1D<TypeParam> inputs(this->num_inputs_, 4);
  Vector1D<TypeParam> expected(this->num_outputs_, 1);
  Vector1D<TypeParam> actual = this->layer_.predict(inputs);
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
  LayerFixture,
  initializationTest, copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  getWeightsRowTest, getWeightsTest,
  predictTest
);

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Layer_Suite, LayerFixture, test_types);
