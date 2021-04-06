#include "test-classes/swarms/clpso_swarm_test.hpp"
#include "../src/common/vector_types.hpp"
#include "../src/common/params/clpso_params.hpp"
#include "../src/common/inertia_weights/linear_decreasing_inertia_weight.hpp"
#include <memory>
#include <climits>
#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::Each;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Le;

template <typename T>
class CLPSO_Swarm_Fixture : public ::testing::Test 
{
protected:
  CLPSO_Swarm_Fixture():
    params_(2, -5, 5, 1.49945, 7, 12,
            std::make_shared<LinearDecreasingInertiaWeight<T, std::size_t>>(0.9, 0.4)),
    test_swarm_(params_)
  {
  }

  ~CLPSO_Swarm_Fixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  clpso_params_t<T, std::size_t> params_;
  CLPSO_Swarm_Test<T, std::size_t> test_swarm_;
};

TYPED_TEST_SUITE_P(CLPSO_Swarm_Fixture);

TYPED_TEST_P(CLPSO_Swarm_Fixture, initializationTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Check main variables
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->test_swarm_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->test_swarm_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->test_swarm_.getCogWeight());
  EXPECT_EQ(this->params_.refreshing_gap, this->test_swarm_.getRefreshingGap());
  EXPECT_EQ(num_dim, this->test_swarm_.getNumDimensions());
  EXPECT_EQ(0, this->test_swarm_.getBestIndex());

  // Check Fi array
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getFis().size());
  for (std::size_t i = 0; i < this->params_.num_particles; ++i)
  {
    EXPECT_EQ(num_dim, this->test_swarm_.getFis()[i].size());
  }

  // Check generated no update times
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getNoUpdateTimes().size());
  EXPECT_THAT(this->test_swarm_.getNoUpdateTimes(), Each(AllOf(Eq(this->params_.refreshing_gap))));

  // Check generated probabilities
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getProbabilities().size());
  EXPECT_THAT(this->test_swarm_.getProbabilities(), Each(AllOf(Ge(0), Le(0.501))));
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, copyConstructorTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Make action (use copy constructor to copy old swarm to new)
  CLPSO_Swarm_Test<TypeParam> test_swarm2(this->test_swarm_);

  // Check that swarms are equal
  EXPECT_EQ(this->test_swarm_.getNumParticles(), test_swarm2.getNumParticles());
  EXPECT_EQ(this->test_swarm_.getBlo(), test_swarm2.getBlo());
  EXPECT_EQ(this->test_swarm_.getBup(), test_swarm2.getBup());
  EXPECT_EQ(this->test_swarm_.getCogWeight(), test_swarm2.getCogWeight());
  EXPECT_EQ(this->test_swarm_.getRefreshingGap(), test_swarm2.getRefreshingGap());
  EXPECT_EQ(this->test_swarm_.getBestIndex(), test_swarm2.getBestIndex());
  EXPECT_EQ(this->test_swarm_.getNumDimensions(), test_swarm2.getNumDimensions());

  EXPECT_EQ(this->test_swarm_.getFis(), test_swarm2.getFis());
  EXPECT_EQ(this->test_swarm_.getProbabilities(), test_swarm2.getProbabilities());
  EXPECT_EQ(this->test_swarm_.getNoUpdateTimes(), test_swarm2.getNoUpdateTimes());
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, copyOperatorTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Create new swarm
  clpso_params_t<TypeParam> new_params(12, -1, 1, 1.4532, 5, 42,
    std::make_shared<LinearDecreasingInertiaWeight<TypeParam>>(0.3, 0.1));
  CLPSO_Swarm_Test<TypeParam> test_swarm2(new_params);
  
  // Make action (copy old swarm to new)
  test_swarm2 = this->test_swarm_;

  // Check that swarms are equal
  EXPECT_EQ(this->test_swarm_.getNumParticles(), test_swarm2.getNumParticles());
  EXPECT_EQ(this->test_swarm_.getBlo(), test_swarm2.getBlo());
  EXPECT_EQ(this->test_swarm_.getBup(), test_swarm2.getBup());
  EXPECT_EQ(this->test_swarm_.getCogWeight(), test_swarm2.getCogWeight());
  EXPECT_EQ(this->test_swarm_.getRefreshingGap(), test_swarm2.getRefreshingGap());
  EXPECT_EQ(this->test_swarm_.getBestIndex(), test_swarm2.getBestIndex());
  EXPECT_EQ(this->test_swarm_.getNumDimensions(), test_swarm2.getNumDimensions());

  EXPECT_EQ(this->test_swarm_.getFis(), test_swarm2.getFis());
  EXPECT_EQ(this->test_swarm_.getProbabilities(), test_swarm2.getProbabilities());
  EXPECT_EQ(this->test_swarm_.getNoUpdateTimes(), test_swarm2.getNoUpdateTimes());
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, copyOperatorSelfTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();
  // Get swarm-specific variables
  Vector2D<std::size_t> fi = this->test_swarm_.getFis();
  Vector1D<TypeParam> pc = this->test_swarm_.getProbabilities();
  Vector1D<std::size_t> no_upd_times = this->test_swarm_.getNoUpdateTimes();

  // Make action (self-copy)
  this->test_swarm_ = this->test_swarm_;

  // Check that swarms are equal
  EXPECT_EQ(this->params_.num_particles, this->test_swarm_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->test_swarm_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->test_swarm_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->test_swarm_.getCogWeight());
  EXPECT_EQ(this->params_.refreshing_gap, this->test_swarm_.getRefreshingGap());
  EXPECT_EQ(num_dim, this->test_swarm_.getNumDimensions());
  EXPECT_EQ(0, this->test_swarm_.getBestIndex());

  EXPECT_EQ(fi, this->test_swarm_.getFis());
  EXPECT_EQ(pc, this->test_swarm_.getProbabilities());
  EXPECT_EQ(no_upd_times, this->test_swarm_.getNoUpdateTimes());
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, goodBetweenBordersTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  // current num particles = 2
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  // Make action and check
  EXPECT_TRUE(this->test_swarm_.delegate_between_borders(0));
  EXPECT_TRUE(this->test_swarm_.delegate_between_borders(1));
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, badBetweenBordersTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  // current num particles = 2
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.getParticles()->x[0] = {2000};

  // Make action and check
  EXPECT_FALSE(this->test_swarm_.delegate_between_borders(0));
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, goodChooseParticlesTest1)
{
  // Create new swarm
  this->params_.num_particles = 3;
  CLPSO_Swarm_Test<TypeParam> test_swarm2(this->params_);

  // Initialize swarm parameters
  this->test_swarm_ = test_swarm2;
  std::size_t num_dim = 1;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Make action (choose particles indexes)
  std::size_t index = 0;
  std::size_t epoch = 0;
  this->test_swarm_.chooseParticlesToLearnFrom(index, epoch);

  // Check that for current particle update time reset
  EXPECT_EQ(0, this->test_swarm_.getNoUpdateTimes()[index]);
}

// Alternative test with guaranteed access
// to choose particles for each dimension
TYPED_TEST_P(CLPSO_Swarm_Fixture, goodChooseParticlesTest2)
{
  // Create new swarm
  this->params_.num_particles = 3;
  CLPSO_Swarm_Test<TypeParam> test_swarm2(this->params_);

  // Initialize swarm parameters
  this->test_swarm_ = test_swarm2;
  std::size_t num_dim = 1;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Correct particles probabilities (only for test)
  this->test_swarm_.getProbabilities() = {1, 1, 1};

  // Make action
  std::size_t index = 0;
  std::size_t epoch = 1;
  this->test_swarm_.chooseParticlesToLearnFrom(index, epoch);

  // Check that for current particle update time reset
  EXPECT_EQ(0, this->test_swarm_.getNoUpdateTimes()[index]);
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, badChooseParticlesTest)
{
  // Prepare data
  // Create new swarm
  this->params_.num_particles = 3;
  CLPSO_Swarm_Test<TypeParam> test_swarm2(this->params_);

  // Initialize swarm parameters
  this->test_swarm_ = test_swarm2;
  std::size_t num_dim = 1;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Set no update times
  std::size_t epoch = 1;
  std::size_t index = 0;
  this->test_swarm_.getNoUpdateTimes()[index] = 1;

  // Make action
  this->test_swarm_.chooseParticlesToLearnFrom(index, epoch);

  // Check that for current particle update time not reset
  EXPECT_NE(0, this->test_swarm_.getNoUpdateTimes()[index]);
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, fiEqualsTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  // num particles = 2
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeFiArray();
  std::size_t index = 1;
  this->test_swarm_.getFis()[index] = {index, index};

  // Make action and check
  EXPECT_TRUE(this->test_swarm_.delegate_fi_particles_equals_to(index));
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, clearUnusedMemoryTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Check that particles are initialized
  EXPECT_NE(nullptr, this->test_swarm_.getParticles());

  // Try to delete allocated memory
  this->test_swarm_.clearUnusedMemory();

  // Check that all dynamic memory free now
  EXPECT_EQ(nullptr, this->test_swarm_.getParticles());
  EXPECT_EQ(0, this->test_swarm_.getProbabilities().size());
  EXPECT_EQ(0, this->test_swarm_.getNoUpdateTimes().size());
  EXPECT_EQ(0, this->test_swarm_.getFis().size());
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, updateParticleTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 1;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Set own values to see changes for 2 particles
  Vector1D<TypeParam> values = {3, 20};
  this->test_swarm_.getParticles()->x = values;
  this->test_swarm_.getParticles()->v = values;

  // Update particle (current best particle index = 0)
  std::size_t update_index = 1;
  TypeParam weight = 2;
  this->test_swarm_.updateParticle(update_index, weight);

  // Check that second particle was updated
  EXPECT_NE(values[update_index * num_dim], this->test_swarm_.getParticles()->x[update_index * num_dim]);
  EXPECT_NE(values[update_index * num_dim], this->test_swarm_.getParticles()->v[update_index * num_dim]);
}

TYPED_TEST_P(CLPSO_Swarm_Fixture, goodUpdateBestParticleStateTest)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Set own function values for 2 particles
  std::size_t update_index = 1;
  Vector1D<TypeParam> f_values = {3, 1};
  Vector1D<TypeParam> fbest_values = {3, 20};
  this->test_swarm_.getParticles()->f_curr = f_values;
  this->test_swarm_.getParticles()->f_best = fbest_values;
  Vector1D<TypeParam> f_best(this->test_swarm_.getParticles()->p.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->p.cbegin() + update_index * num_dim + num_dim);
  TypeParam weight = 5;

  // Check that for current particles state updated
  EXPECT_TRUE(this->test_swarm_.tryToUpdateBestParticleState(update_index, weight));

  Vector1D<TypeParam> curr_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  EXPECT_EQ(f_best, curr_x);
}

// Test bad particles state update due to 'x' value borders limitation
TYPED_TEST_P(CLPSO_Swarm_Fixture, badUpdateBestParticleStateTest1)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();

  TypeParam weight = 5;
  std::size_t update_index = 1;

  // Correct particle's 'x' placement entry value to fail test (like
  // calculated after updateParticles() procedure)
  this->test_swarm_.getParticles()->x[update_index * num_dim] = this->params_.b_up + 1;
  Vector1D<TypeParam> old_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                            this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);

  // Check that for current particles state didn't update
  EXPECT_FALSE(this->test_swarm_.tryToUpdateBestParticleState(update_index, weight));

  Vector1D<TypeParam> curr_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  EXPECT_EQ(old_x, curr_x);
}

// Test bad particles state update due to bad new function value
TYPED_TEST_P(CLPSO_Swarm_Fixture, badUpdateBestParticleStateTest2)
{
  // Initialize swarm parameters
  std::size_t num_dim = 2;
  this->test_swarm_.setNumDimensions(num_dim);
  this->test_swarm_.initializeSwarm();
  this->test_swarm_.initializeFiArray();
  this->test_swarm_.initializeNoUpdateTimes();
  this->test_swarm_.generateProbabilities();

  // Set own function values for 2 particles
  std::size_t update_index = 1;
  Vector1D<TypeParam> f_values = {3, 20};
  Vector1D<TypeParam> fbest_values = {3, 1};
  this->test_swarm_.getParticles()->f_curr = f_values;
  this->test_swarm_.getParticles()->f_best = fbest_values;
  Vector1D<TypeParam> old_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                            this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  TypeParam weight = 5;

  // Check that for current particles state didn't update
  EXPECT_FALSE(this->test_swarm_.tryToUpdateBestParticleState(update_index, weight));

  Vector1D<TypeParam> curr_x(this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim,
                             this->test_swarm_.getParticles()->x.cbegin() + update_index * num_dim + num_dim);
  EXPECT_EQ(old_x, curr_x);
}

REGISTER_TYPED_TEST_SUITE_P
(
  CLPSO_Swarm_Fixture,
  initializationTest, copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  goodBetweenBordersTest, badBetweenBordersTest,
  fiEqualsTest,
  goodChooseParticlesTest1,
  goodChooseParticlesTest2,
  badChooseParticlesTest,
  clearUnusedMemoryTest,
  updateParticleTest,
  goodUpdateBestParticleStateTest,
  badUpdateBestParticleStateTest1,
  badUpdateBestParticleStateTest2
 );

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(CLPSO_Swarm_Suite, CLPSO_Swarm_Fixture, test_types);
