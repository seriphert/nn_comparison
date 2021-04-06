#include "../src/layers/backprop/backprop_layer.hpp"
#include "../src/common/activation_functions/sigmoid.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>

template <typename T>
class Backprop_Layer_Fixture : public ::testing::Test
{
protected:
  Backprop_Layer_Fixture():
    enableBias_(true),
    num_inputs_(4),
    num_outputs_(5),
    weights_size_(num_inputs_ + ((enableBias_) ? 1 : 0)),
    layer_(num_inputs_, num_outputs_, std::make_shared<Sigmoid<T>>(), enableBias_)
  {
  }

  ~Backprop_Layer_Fixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  bool enableBias_;
  std::size_t num_inputs_;
  std::size_t num_outputs_;
  std::size_t weights_size_;
  BackPropLayer<T, std::size_t> layer_;
};

TYPED_TEST_SUITE_P(Backprop_Layer_Fixture);

TYPED_TEST_P(Backprop_Layer_Fixture, initializationTest)
{
  // Check that variables defined for layer in fixture
  // are similar with initial fixture's variables
  EXPECT_EQ(this->num_inputs_, this->layer_.getNumInputs());
  EXPECT_EQ(this->num_outputs_, this->layer_.getNumOutputs());
  EXPECT_EQ(this->weights_size_, this->layer_.getWeights().size());
  
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(this->num_outputs_, this->layer_.getWeightsRow(i).size());
  }

  EXPECT_EQ(0, this->layer_.getInputs().size());
  EXPECT_EQ(0, this->layer_.getDeltas().size());
}

TYPED_TEST_P(Backprop_Layer_Fixture, copyConstructorTest)
{
  // Copy fixture's layer to new
  BackPropLayer<TypeParam> layer(this->layer_);

  // Check that copied layer variables equal to fixture layer
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights().size(), layer.getWeights().size());
  
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(this->layer_.getWeightsRow(i), layer.getWeightsRow(i));
  }

  EXPECT_EQ(this->layer_.getInputs(), layer.getInputs());
  EXPECT_EQ(this->layer_.getDeltas(), layer.getDeltas());
}

TYPED_TEST_P(Backprop_Layer_Fixture, copyOperatorTest)
{
  // Define new layer
  BackPropLayer<TypeParam> layer = {2, 2, std::make_shared<Sigmoid<TypeParam>>(), false};

  // Use copy operator to clone fixture's layer
  // in created layer
  layer = this->layer_;

  // Check that copied layer variables equal to fixture's layer variables
  EXPECT_EQ(this->layer_.getNumInputs(), layer.getNumInputs());
  EXPECT_EQ(this->layer_.getNumOutputs(), layer.getNumOutputs());
  EXPECT_EQ(this->layer_.getWeights().size(), layer.getWeights().size());
  
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    EXPECT_EQ(this->layer_.getWeightsRow(i), layer.getWeightsRow(i));
  }

  EXPECT_EQ(this->layer_.getInputs(), layer.getInputs());
  EXPECT_EQ(this->layer_.getDeltas(), layer.getDeltas());
}

TYPED_TEST_P(Backprop_Layer_Fixture, copyOperatorSelfTest)
{
  // Define fixture's variables
  std::size_t num_inputs = this->num_inputs_;
  std::size_t num_outputs = this->num_outputs_;
  Vector2D<TypeParam> weights = this->layer_.getWeights();
  Vector1D<TypeParam> inputs = this->layer_.getInputs();
  Vector1D<TypeParam> deltas = this->layer_.getDeltas();

  // Try to copy layer with itself
  this->layer_ = this->layer_;

  // Check that layer's variables are same
  EXPECT_EQ(num_inputs, this->layer_.getNumInputs());
  EXPECT_EQ(num_outputs, this->layer_.getNumOutputs());
  EXPECT_EQ(weights, this->layer_.getWeights());
  EXPECT_EQ(inputs, this->layer_.getInputs());
  EXPECT_EQ(deltas, this->layer_.getDeltas());
}

TYPED_TEST_P(Backprop_Layer_Fixture, goodInputsTest)
{
  // Set inputs to layer
  Vector1D<TypeParam> expected(this->num_inputs_, 1);  

  // Check that inputs vector's size equal to layer input size 
  EXPECT_NO_THROW(this->layer_.setInputs(expected));

  // Get inputs from layer
  Vector1D<TypeParam> actual = this->layer_.getInputs();

  // Check that expected layer inputs equal to actual
  EXPECT_EQ(expected, actual);
}

TYPED_TEST_P(Backprop_Layer_Fixture, badInputsTest)
{
  // Define test variables with wrong layer input size
  std::size_t wrong_size = this->num_inputs_ - 1;
  Vector1D<TypeParam> inputs(wrong_size, 1);

  // Check that inputs cannot be set
  EXPECT_THROW(this->layer_.setInputs(inputs), std::invalid_argument);
}

TYPED_TEST_P(Backprop_Layer_Fixture, goodCalculateDeltasTest)
{
  // Set weights to layer
  Vector1D<TypeParam> weights(this->weights_size_, 1);
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    this->layer_.getWeightsRow(i) = weights;
  }
  // Set inputs to layer
  Vector1D<TypeParam> inputs(this->num_inputs_, 2);
  this->layer_.setInputs(inputs);

  // Define deltas for SGD method and expected outputs
  Vector1D<TypeParam> prev_deltas(this->num_outputs_, 1);
  Vector1D<TypeParam> expected(this->num_inputs_, -10);

  // Check that deltas size equals to layer outputs size
  // and actual output delta values equals to expected 
  EXPECT_NO_THROW(this->layer_.calculateDeltas(prev_deltas));
  Vector1D<TypeParam> actual = this->layer_.getDeltas();
  EXPECT_EQ(expected, actual);
}

TYPED_TEST_P(Backprop_Layer_Fixture, badCalculateDeltasTest)
{
  // As in previous test, but with wrong layer outputs size
  std::size_t wrong_size = this->num_outputs_ + 1;
  Vector1D<TypeParam> prev_deltas(wrong_size, 1);

  // Check that deltas cannot be calculated due to their wrong size
  EXPECT_THROW(this->layer_.calculateDeltas(prev_deltas), std::invalid_argument);
}

TYPED_TEST_P(Backprop_Layer_Fixture, applyDerivativeToValuesTest)
{
  // Define test variables
  Vector1D<TypeParam> expected = {2};
  Vector1D<TypeParam> actual = expected;

  // Calculate expected derivatives
  Sigmoid<TypeParam>{}.derivative(expected);

  // Calculate actual derivatives
  this->layer_.applyDerivativeToValues(actual);

  // Check that derivatives for expected
  // and actual are equal
  EXPECT_EQ(expected, actual);
}

TYPED_TEST_P(Backprop_Layer_Fixture, goodRecalculateWeightsTest)
{
  // Set weights to layer
  Vector1D<TypeParam> weights(this->weights_size_, 1);
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    this->layer_.getWeightsRow(i) = weights;
  }
  // Set inputs to layer
  Vector1D<TypeParam> inputs(this->num_inputs_, 1);
  this->layer_.setInputs(inputs);

  // Define SGD variables
  TypeParam learn_rate = 0.5;
  TypeParam momentum = 0.1;
  Vector1D<TypeParam> deltas(this->num_outputs_, -10);
  
  // Calculated expected weights
  Vector2D<TypeParam> expected(this->weights_size_);
  for (std::size_t i = 0; i < this->weights_size_; ++i)
  {
    expected[i].assign(this->num_outputs_, -4);
  }

  // Check that deltas size equal to layer outputs size 
  EXPECT_NO_THROW(this->layer_.recalculateWeights(learn_rate, momentum, deltas));
  
  // Calculate new weights
  Vector2D<TypeParam> actual = this->layer_.getWeights();

  // Check that actual weights equal to expected
  EXPECT_EQ(expected, actual);
}

TYPED_TEST_P(Backprop_Layer_Fixture, badRecalculateWeightsTest)
{
  // Define SGD variables
  TypeParam learn_rate = 0.5;
  TypeParam momentum = 0.1;
  // Use wrong size instead of correct
  std::size_t wrong_size = this->num_outputs_ + 1;
  // Define deltas
  Vector1D<TypeParam> deltas(wrong_size, -10);
  
  // Check that weights cannot be recalculated due to
  // invalid deltas size
  EXPECT_THROW(this->layer_.recalculateWeights(learn_rate, momentum, deltas),
               std::invalid_argument);
}

REGISTER_TYPED_TEST_SUITE_P
(
  Backprop_Layer_Fixture,
  initializationTest, copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  goodInputsTest, badInputsTest,
  goodCalculateDeltasTest, badCalculateDeltasTest,
  applyDerivativeToValuesTest,
  goodRecalculateWeightsTest, badRecalculateWeightsTest
);

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(BackPropLayer_Suite, Backprop_Layer_Fixture, test_types);
