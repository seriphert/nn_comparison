#include "mocks/swarm_mock.hpp"
#include "../src/common/vector_types.hpp"
#include "../src/common/params/base_pso_params.hpp"
#include "../src/common/inertia_weights/linear_decreasing_inertia_weight.hpp"
#include <memory>
#include <climits>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::Each;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Le;

template <typename T>
class SwarmFixture : public ::testing::Test 
{
protected:
  SwarmFixture():
    params_(2, -5, 5, 1.49945, 12,
            std::make_shared<LinearDecreasingInertiaWeight<T, std::size_t>>(0.9, 0.4)),
    mock_(params_)
  {
  }

  ~SwarmFixture() override = default;

  void SetUp() override {}

  void TearDown() override {}

  base_pso_params_t<T, std::size_t> params_;
  SwarmMock<T> mock_;
};

TYPED_TEST_SUITE_P(SwarmFixture);

TYPED_TEST_P(SwarmFixture, initializationTest)
{
  // Check that variables defined for swarm in fixture are
  // similar with initial fixture's variables
  EXPECT_EQ(this->params_.num_particles, this->mock_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->mock_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->mock_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->mock_.getCogWeight());
  EXPECT_EQ(0, this->mock_.getBestIndex());
  EXPECT_EQ(0, this->mock_.getNumDimensions());
}

TYPED_TEST_P(SwarmFixture, copyConstructorTest)
{
  // Copy fixture's swarm to new swarm
  SwarmMock<TypeParam> mock2(this->mock_);

  // Check that copied swarm variables equal to fixture swarm
  EXPECT_EQ(this->mock_.getNumParticles(), mock2.getNumParticles());
  EXPECT_EQ(this->mock_.getBlo(), mock2.getBlo());
  EXPECT_EQ(this->mock_.getBup(), mock2.getBup());
  EXPECT_EQ(this->mock_.getCogWeight(), mock2.getCogWeight());
  EXPECT_EQ(this->mock_.getBestIndex(), mock2.getBestIndex());
  EXPECT_EQ(this->mock_.getNumDimensions(), mock2.getNumDimensions());
}

TYPED_TEST_P(SwarmFixture, copyOperatorTest)
{
  // Define new swarm
  base_pso_params_t<TypeParam> new_params(12, -1, 1, 1.4532, 42,
    std::make_shared<LinearDecreasingInertiaWeight<TypeParam>>(0.3, 0.1));
  SwarmMock<TypeParam> mock2(new_params);

  // Use copy operator to clone fixture's layer
  // in created layer
  mock2 = this->mock_;

  // Check that copied swarm variables equal to fixture's swarm variables
  EXPECT_EQ(this->mock_.getNumParticles(), mock2.getNumParticles());
  EXPECT_EQ(this->mock_.getBlo(), mock2.getBlo());
  EXPECT_EQ(this->mock_.getBup(), mock2.getBup());
  EXPECT_EQ(this->mock_.getCogWeight(), mock2.getCogWeight());
  EXPECT_EQ(this->mock_.getBestIndex(), mock2.getBestIndex());
  EXPECT_EQ(this->mock_.getNumDimensions(), mock2.getNumDimensions());
}

TYPED_TEST_P(SwarmFixture, copyOperatorSelfTest)
{
  // Try to copy layer with itself
  this->mock_ = this->mock_;

  // Check that no updates in swarm
  EXPECT_EQ(this->params_.num_particles, this->mock_.getNumParticles());
  EXPECT_EQ(this->params_.b_lo, this->mock_.getBlo());
  EXPECT_EQ(this->params_.b_up, this->mock_.getBup());
  EXPECT_EQ(this->params_.cog_weight, this->mock_.getCogWeight());
  EXPECT_EQ(0, this->mock_.getBestIndex());
  EXPECT_EQ(0, this->mock_.getNumDimensions());
}

TYPED_TEST_P(SwarmFixture, setNumDimensionsTest)
{
  // Set particle's dimensions
  std::size_t new_dim = 2;
  this->mock_.setNumDimensions(new_dim);

  // Check that actual dimensions number
  // equals to defined earlier
  EXPECT_EQ(new_dim, this->mock_.getNumDimensions());
}

TYPED_TEST_P(SwarmFixture, initializeSwarmTest)
{
  // Get particles reference pointer and reset for initialization
  std::shared_ptr<particles_t<TypeParam>> & particles = this->mock_.getParticles();
  particles.reset();

  // Set number of dimensions for each particle to swarm
  std::size_t num_dim = 2;
  this->mock_.setNumDimensions(num_dim);

  // Set swarm initialization parametes
  std::size_t particles_num_func_values = this->params_.num_particles;
  std::size_t particles_num_dimensions = this->params_.num_particles * num_dim;
  TypeParam particle_func_value = std::numeric_limits<TypeParam>::infinity();
  TypeParam v_max = std::abs(this->params_.b_up - this->params_.b_lo);

  // Try to iniailize swarm
  this->mock_.initializeSwarm();
  
  // Check initialized particles entries sizes
  EXPECT_EQ(particles->x.size(), particles_num_dimensions);
  EXPECT_EQ(particles->p.size(), particles_num_dimensions);
  EXPECT_EQ(particles->v.size(), particles_num_dimensions);
  EXPECT_EQ(particles->f_curr.size(), particles_num_func_values);
  EXPECT_EQ(particles->f_best.size(), particles_num_func_values);

  // Check that all values equals / in range
  EXPECT_THAT(particles->x, Each(AllOf(Ge(this->params_.b_lo), Le(this->params_.b_up))));
  EXPECT_THAT(particles->p, Each(AllOf(Ge(this->params_.b_lo), Le(this->params_.b_up))));
  EXPECT_THAT(particles->v, Each(AllOf(Ge(-v_max), Le(v_max))));
  EXPECT_THAT(particles->f_curr, Each(AllOf(Eq(particle_func_value))));
  EXPECT_THAT(particles->f_best, Each(AllOf(Eq(particle_func_value))));
}

TYPED_TEST_P(SwarmFixture, successfulUpdateBestParticleTest)
{
  this->mock_.initializeSwarm();

  // Get particles reference pointer and reset for initialization
  std::shared_ptr<particles_t<TypeParam>> & particles = this->mock_.getParticles();
  // Current particles number - 2 (set in fixture)
  particles->f_best = {20, 5};

  // Current particles best index = 0 (by default)
  // Try to set new best index (5 < 20)
  std::size_t new_best_index = 1;
  this->mock_.tryToUpdateBestParticle(new_best_index);

  // Check that best index changed
  EXPECT_EQ(new_best_index, this->mock_.getBestIndex());
}

TYPED_TEST_P(SwarmFixture, unsuccessfulUpdateBestParticleTest)
{
  this->mock_.initializeSwarm();

  // Get particles reference pointer and reset for initialization
  std::shared_ptr<particles_t<TypeParam>> & particles = this->mock_.getParticles();
  // Current particles number - 2 (set in fixture)
  particles->f_best = {5, 20};

  // Current particles best index = 0 (by default)
  // Try to set new best index (5 < 20)
  std::size_t curr_best_index = 0;
  std::size_t new_best_index = 1;
  this->mock_.tryToUpdateBestParticle(new_best_index);

  // Check that best index unchanged
  EXPECT_EQ(curr_best_index, this->mock_.getBestIndex());
}

TYPED_TEST_P(SwarmFixture, getStartWeightsIteratorTest)
{
  // Initialize swarm
  this->mock_.initializeSwarm();

  // Get particles reference pointer and reset for initialization
  std::shared_ptr<particles_t<TypeParam>> & particles = this->mock_.getParticles();
 
  typename Vector1D<TypeParam>::const_iterator start_iter
      = this->mock_.getStartWeightsIterator(0);

  // Check that start iterator begins with 0 particle's 'x' start position 
  EXPECT_EQ(particles->x.cbegin(), start_iter);
}

REGISTER_TYPED_TEST_SUITE_P
(
  SwarmFixture,
  initializationTest, copyConstructorTest,
  copyOperatorTest, copyOperatorSelfTest,
  setNumDimensionsTest,
  initializeSwarmTest,
  successfulUpdateBestParticleTest,
  unsuccessfulUpdateBestParticleTest,
  getStartWeightsIteratorTest
);

using test_types = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Swarm_Suite, SwarmFixture, test_types);
