using MonteCarloX
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf


""" Testing if sampling the uniform case correctly"""
function test_poisson_constant()
  LambdaMaxs = [1.0, 2.0] # testing for two different max-sample-rates
  Lambda(t) = 1.0
  pass = true
  
  for LambdaMax in LambdaMaxs
    rng = MersenneTwister(1000)
    nSamples = 1000
    samples = zeros(nSamples)

    t0 = 0    
    for i in 1:nSamples
      t0 += InhomogeneousPoissonProcess.next_event_time(t -> Lambda(t + t0), LambdaMax, rng)
      samples[i] = t0 % 1
    end

    test = HypothesisTests.ExactOneSampleKSTest(samples, Uniform(0,1))
    pass &= pvalue(test) > 0.05
  end

  return pass
end

""" Testing if sampling correctly from a shifted sine-wave distribution."""
function test_poisson_sin_wave()
  LambdaMaxs = [2.0, 3.0] # testing for two different max-sample-rates
  Lambda(t) = sin(t) + 1.0
  pass = true

  for LambdaMax in LambdaMaxs
    rng = MersenneTwister(1000)
    nSamples = 1000
    samples = zeros(nSamples)

    t0 = 0    
    for i in 1:nSamples
      t0 += InhomogeneousPoissonProcess.next_event_time(t -> Lambda(t + t0), LambdaMax, rng)
      samples[i] = t0 % (2 * pi)
    end

    test = HypothesisTests.ExactOneSampleKSTest(samples, SinDistribution())
    pass &= pvalue(test) > 0.05
  end

  return pass
end

""" Definition of the custom sine-wave distribution for hypothesis testing."""
struct SinDistribution <: ContinuousUnivariateDistribution end

function Distributions.minimum(d::SinDistribution)
    0.0
end

function Distributions.maximum(d::SinDistribution)
    2*pi
end

function pdf(d::SinDistribution, x::Real)
    insupport(d, x) ? (sin(x)+1) / (2*pi) : 0.0
end

function cdf(d::SinDistribution, x::Real)
    insupport(d, x) ? (x - cos(x) + 1.0) / (2*pi) : Float64(x > 0)
end



