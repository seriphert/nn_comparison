#include "test-classes/swarms/pso_swarm_test.hpp"
#include "../src/common/vector_types.hpp"
#include "../src/common/params/pso_params.hpp"
#include "../src/common/inertia_weights/linear_decreasing_inertia_weight.hpp"
#include <memory>
#include <climits>
#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

template <typename T>
class PSO_Swarm_Fixture : public ::testing::Test 
{
protected:
  PSO_Swarm_Fixture() :
    params_(2, -5, 5, 1.49945, 1.49945, 12,
            std::make_shared<LinearDecreasingInertiaWeight<T, std::size_t>>(0.9, 0.4)),
    test_swarm_(params_)
  {
  }

  ~PSO_Swarm_Fixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  pso_params_t<T, std::size_t> params_;
  PSO_Swarm_Test<T, std::size_t> test_swarm_;
};

TYPED_TEST_SUITE_P(PSO_Swarm_Fixture);

TYPED_TEST_P(PSO_Swarm_Fixture, initializationTest)
{
  // Make action
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  // Check
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->test_swarm_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->test_swarm_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->test_swarm_.getCogWeight());
  EXPECT_EQ(this->params_.soc_weight, this->test_swarm_.getSocWeight());
  EXPECT_EQ(num_dim, this->test_swarm_.getNumDimensions());
  EXPECT_EQ(0, this->test_swarm_.getBestIndex());
  // For initializeSwarm() have test in 'test-swarm.cpp' file
  // just check that partices pointer not null
  EXPECT_NE(nullptr, this->test_swarm_.getParticles());
}

TYPED_TEST_P(PSO_Swarm_Fixture, copyConstructorTest)
{
  // Make action
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  PSO_Swarm_Test<TypeParam> test_swarm2(this->test_swarm_);

  // Check
  EXPECT_EQ(this->test_swarm_.getNumParticles(), test_swarm2.getNumParticles());
  EXPECT_EQ(this->test_swarm_.getBlo(), test_swarm2.getBlo());
  EXPECT_EQ(this->test_swarm_.getBup(), test_swarm2.getBup());
  EXPECT_EQ(this->test_swarm_.getCogWeight(), test_swarm2.getCogWeight());
  EXPECT_EQ(this->test_swarm_.getSocWeight(), test_swarm2.getSocWeight());
  EXPECT_EQ(this->test_swarm_.getBestIndex(), test_swarm2.getBestIndex());
  EXPECT_EQ(this->test_swarm_.getNumDimensions(), test_swarm2.getNumDimensions());
  EXPECT_NE(nullptr, test_swarm2.getParticles());
}

TYPED_TEST_P(PSO_Swarm_Fixture, copyOperatorTest)
{
  // Make action
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  pso_params_t<TypeParam> new_params(12, -1, 1, 1.4532, 1.4532, 42,
    std::make_shared<LinearDecreasingInertiaWeight<TypeParam>>(0.3, 0.1));
  PSO_Swarm_Test<TypeParam> test_swarm2(new_params);
  test_swarm2 = this->test_swarm_;

  // Check
  EXPECT_EQ(this->test_swarm_.getNumParticles(), test_swarm2.getNumParticles());
  EXPECT_EQ(this->test_swarm_.getBlo(), test_swarm2.getBlo());
  EXPECT_EQ(this->test_swarm_.getBup(), test_swarm2.getBup());
  EXPECT_EQ(this->test_swarm_.getCogWeight(), test_swarm2.getCogWeight());
  EXPECT_EQ(this->test_swarm_.getSocWeight(), test_swarm2.getSocWeight());
  EXPECT_EQ(this->test_swarm_.getBestIndex(), test_swarm2.getBestIndex());
  EXPECT_EQ(this->test_swarm_.getNumDimensions(), test_swarm2.getNumDimensions());
  EXPECT_NE(nullptr, test_swarm2.getParticles());
}

TYPED_TEST_P(PSO_Swarm_Fixture, copyOperatorSelfTest)
{
  // Make action
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_ = this->test_swarm_;

  // Check
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->test_swarm_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->test_swarm_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->test_swarm_.getCogWeight());
  EXPECT_EQ(this->params_.soc_weight, this->test_swarm_.getSocWeight());
  EXPECT_EQ(num_dim, this->test_swarm_.getNumDimensions());
  EXPECT_EQ(0, this->test_swarm_.getBestIndex());
  EXPECT_NE(nullptr, this->test_swarm_.getParticles());
}

TYPED_TEST_P(PSO_Swarm_Fixture, updateParticleTest)
{
  // Make actions:
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  // Set own values to see changes for 2 particles
  Vector1D<TypeParam> values = {3, 4, 20, 1};
  this->test_swarm_.getParticles()->x = values;
  this->test_swarm_.getParticles()->v = values;

  // Update particle (current best particle index = 0)
  std::size_t update_index = 1;
  TypeParam weight = 2;
  this->test_swarm_.updateParticle(update_index, weight);

  // Check that second particle was updated
  EXPECT_NE(values[update_index * num_dim], this->test_swarm_.getParticles()->x[update_index * num_dim]);
  EXPECT_NE(values[update_index * num_dim], this->test_swarm_.getParticles()->v[update_index * num_dim]);
  EXPECT_NE(values[update_index * num_dim + 1], this->test_swarm_.getParticles()->x[update_index * num_dim + 1]);
  EXPECT_NE(values[update_index * num_dim + 1], this->test_swarm_.getParticles()->v[update_index * num_dim + 1]);
}

TYPED_TEST_P(PSO_Swarm_Fixture, successfulUpdateBestParticleStateTest)
{
  // Initialize swarm
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  // Set own function values for 2 particles
  std::size_t update_index = 1;
  Vector1D<TypeParam> f_values = {3, 1};
  Vector1D<TypeParam> fbest_values = {3, 20};
  this->test_swarm_.getParticles()->f_curr = f_values;
  this->test_swarm_.getParticles()->f_best = fbest_values;
  Vector1D<TypeParam> curr_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  TypeParam weight = 5;

  // Check that particle's best position changed
  EXPECT_TRUE(this->test_swarm_.tryToUpdateBestParticleState(update_index, weight));
  Vector1D<TypeParam> curr_p(this->test_swarm_.getParticles()->p.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->p.cbegin() + update_index * num_dim + num_dim);
  EXPECT_EQ(curr_x, curr_p);
}

TYPED_TEST_P(PSO_Swarm_Fixture, unsuccessfulUpdateBestParticleStateTest)
{
  // Initialize swarm
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  // Set own function values for 2 particles
  std::size_t update_index = 1;
  Vector1D<TypeParam> f_values = {3, 20};
  Vector1D<TypeParam> fbest_values = {3, 1};
  this->test_swarm_.getParticles()->f_curr = f_values;
  this->test_swarm_.getParticles()->f_best = fbest_values;

  Vector1D<TypeParam> curr_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  TypeParam weight = 5;

  // Check that particle's best position unchanged
  EXPECT_FALSE(this->test_swarm_.tryToUpdateBestParticleState(update_index, weight));
  EXPECT_NE(this->test_swarm_.getParticles()->f_best[update_index], f_values[update_index]);
}

TYPED_TEST_P(PSO_Swarm_Fixture, clearUnusedMemoryTest)
{
  // Initialize swarm variables
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  // Check that particles initialized
  EXPECT_NE(nullptr, this->test_swarm_.getParticles());

  // Try to free allocated memory
  this->test_swarm_.clearUnusedMemory();

  // Check that alocated memory free now
  EXPECT_EQ(nullptr, this->test_swarm_.getParticles());
}

REGISTER_TYPED_TEST_SUITE_P
(
  PSO_Swarm_Fixture,
  initializationTest, copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  updateParticleTest,
  clearUnusedMemoryTest,
  successfulUpdateBestParticleStateTest,
  unsuccessfulUpdateBestParticleStateTest
);

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(PSO_Swarm_Suite, PSO_Swarm_Fixture, test_types);
