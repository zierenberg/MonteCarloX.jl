using MonteCarloX
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf


""" Testing if sampling single event correctly"""
function test_poisson_single()
    LambdaMaxs = [10.0, 20.0] # testing for two different max-sample-rates
    Lambda(t) = 10.0
    pass = true

    
    # Regular sampler
    # alg = InhomogeneousPoisson()
    for LambdaMax in LambdaMaxs
        rng = MersenneTwister(1000)
        nSamples = 1000
        samples = zeros(nSamples)

        for i in 1:nSamples
            t0 = next_time(rng, Lambda, LambdaMax)
            samples[i] = t0
        end

        test = HypothesisTests.ExactOneSampleKSTest(samples, Exponential(1.0 / 10.0))
        pass &= pvalue(test) > 0.05
    end

    return pass
end

""" Testing if sampling the uniform case correctly"""
function test_poisson_constant()
    LambdaMaxs = [1.0, 2.0] # testing for two different max-sample-rates
    Lambda(t) = 1.0
    pass = true
    
    # alg = InhomogeneousPoisson()
    for LambdaMax in LambdaMaxs
        rng = MersenneTwister(1000)
        nSamples = 1000
        samples = zeros(nSamples)

        t0 = 0        
        for i in 1:nSamples
            t0 += next_time(rng, t->Lambda(t + t0), LambdaMax)
            samples[i] = t0 % 1
        end

        test = HypothesisTests.ExactOneSampleKSTest(samples, Uniform(0, 1))
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
            t0 += next_time(rng, t->Lambda(t + t0), LambdaMax)
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
    2 * pi
end

function pdf(d::SinDistribution, x::Real)
    insupport(d, x) ? (sin(x) + 1) / (2 * pi) : 0.0
end

function cdf(d::SinDistribution, x::Real)
    insupport(d, x) ? (x - cos(x) + 1.0) / (2 * pi) : Float64(x > 0)
end




### from Claude
using Test

@testset "PoissonProcess Tests" begin
    
    @testset "Constructor Tests" begin
        # Single channel with default RNG
        pp1 = PoissonProcess(5.0)
        @test pp1.rate == 5.0
        @test pp1.rng isa AbstractRNG
        
        # Single channel with custom RNG
        rng = MersenneTwister(123)
        pp2 = PoissonProcess(5; rng=rng)
        @test pp2.rate == 5.0
        @test pp2.rng === rng
        
        # Multi-channel
        pp3 = PoissonProcess([1.0, 2.0, 3.0])
        @test pp3.rate == [1.0, 2.0, 3.0]
        @test pp3.rate isa Vector{Float64}
        
        # Integer gets converted to float
        pp4 = PoissonProcess(5)
        @test pp4.rate isa Float64
    end
    
    @testset "InhomogeneousPoissonProcess Constructor" begin
        rate_func(t) = 10.0 * exp(-0.1 * t)
        
        ipp1 = InhomogeneousPoissonProcess(rate_func, 10.0)
        @test ipp1.rate(0.0) == 10.0
        @test ipp1.generation_rate == 10.0
        
        # With custom RNG
        rng = MersenneTwister(456)
        ipp2 = InhomogeneousPoissonProcess(rate_func, 10.0; rng=rng)
        @test ipp2.rng === rng
        
        # Multi-channel inhomogeneous
        rate_func_multi(t) = [5.0 * exp(-0.1 * t), 3.0 * exp(-0.05 * t)]
        ipp3 = InhomogeneousPoissonProcess(rate_func_multi, [5.0, 3.0])
        @test length(ipp3.rate(0.0)) == 2
    end
    
    # @testset "Homogeneous Process - Event Counting" begin
    #     rng = MersenneTwister(42)
    #     rate = 10.0
    #     pp = PoissonProcess(rate; rng=rng)
        
    #     # Count events
    #     events = Int[]
    #     times = Float64[]
    #     update! = (event, t) -> begin
    #         push!(events, event)
    #         push!(times, t)
    #     end
        
    #     t_final = 100.0
    #     advance!(pp, t_final; update! = update!)
        
    #     # Statistical test: expected number of events ≈ rate * t_final
    #     expected = rate * t_final
    #     @test length(events) > 0
    #     @test abs(length(events) - expected) < 5 * sqrt(expected)  # ~5σ tolerance
        
    #     # All events should be 0 for single channel
    #     @test all(e -> e == 0, events)
        
    #     # Times should be sorted and within bounds
    #     @test issorted(times)
    #     @test all(t -> 0 <= t <= t_final, times)
    # end
    
    # @testset "Multi-channel Homogeneous Process" begin
    #     rng = MersenneTwister(789)
    #     rates = [5.0, 10.0, 15.0]
    #     pp = PoissonProcess(rates; rng=rng)
        
    #     events = Int[]
    #     update! = (event, t) -> push!(events, event)
        
    #     advance!(pp, 100.0; update! = update!)
        
    #     # Should have events from all channels
    #     @test minimum(events) >= 1
    #     @test maximum(events) <= 3
        
    #     # Rough check: channel 3 should have most events (highest rate)
    #     counts = [count(==(i), events) for i in 1:3]
    #     @test counts[3] > counts[1]  # Channel 3 > Channel 1
    # end
    
    # @testset "Inhomogeneous Process - Decaying Rate" begin
    #     rng = MersenneTwister(101)
        
    #     # Exponentially decaying rate
    #     rate_func(t) = 20.0 * exp(-0.1 * t)
    #     generation_rate = 20.0
        
    #     ipp = InhomogeneousPoissonProcess(rate_func, generation_rate; rng=rng)
        
    #     times = Float64[]
    #     update! = (event, t) -> push!(times, t)
        
    #     advance!(ipp, 50.0; update! = update!)
        
    #     # More events should occur early (higher rate)
    #     early_events = count(t -> t < 10.0, times)
    #     late_events = count(t -> t >= 40.0, times)
    #     @test early_events > late_events
        
    #     # Times should be sorted
    #     @test issorted(times)
    # end
    
    # @testset "Inhomogeneous Process - Sinusoidal Rate" begin
    #     rng = MersenneTwister(202)
        
    #     # Periodic rate function
    #     rate_func(t) = 5.0 + 5.0 * sin(2π * t / 10.0)
    #     generation_rate = 10.0
        
    #     ipp = InhomogeneousPoissonProcess(rate_func, generation_rate; rng=rng)
        
    #     times = Float64[]
    #     update! = (event, t) -> push!(times, t)
        
    #     advance!(ipp, 100.0; update! = update!)
        
    #     @test length(times) > 0
    #     @test issorted(times)
    # end
    
    # @testset "Zero Rate Behavior" begin
    #     # Zero rate should return Inf
    #     pp_zero = PoissonProcess(0.0)
    #     t = advance!(pp_zero, 100.0)
    #     @test t == Inf
        
    #     # Zero generation rate for inhomogeneous
    #     ipp_zero = InhomogeneousPoissonProcess(t -> 0.0, 0.0)
    #     t = advance!(ipp_zero, 100.0)
    #     @test t == Inf
    # end
    
    # @testset "Custom t0" begin
    #     rng = MersenneTwister(303)
    #     pp = PoissonProcess(10.0; rng=rng)
        
    #     times = Float64[]
    #     update! = (event, t) -> push!(times, t)
        
    #     t0 = 50.0
    #     advance!(pp, 100.0; t0=t0, update! = update!)
        
    #     # All events should be after t0
    #     @test all(t -> t >= t0, times)
    # end
    
    # @testset "No Update Callback" begin
    #     # Should work without error when update! is not provided
    #     pp = PoissonProcess(10.0)
    #     t = advance!(pp, 10.0)
    #     @test t isa Float64
        
    #     ipp = InhomogeneousPoissonProcess(t -> 5.0, 10.0)
    #     t = advance!(ipp, 10.0)
    #     @test t isa Float64
    # end
    
    # @testset "Reproducibility with RNG" begin
    #     # Same seed should give same results
    #     events1 = Int[]
    #     times1 = Float64[]
    #     pp1 = PoissonProcess(10.0; rng=MersenneTwister(999))
    #     advance!(pp1, 50.0; update! = (pp1, e, t) -> (push!(events1, e); push!(times1, t)))
        
    #     events2 = Int[]
    #     times2 = Float64[]
    #     pp2 = PoissonProcess(10.0; rng=MersenneTwister(999))
    #     advance!(pp2, 50.0; update! = (pp2, e, t) -> (push!(events2, e); push!(times2, t)))
        
    #     @test events1 == events2
    #     @test times1 == times2
    # end
    
    # @testset "Type Stability" begin
    #     pp = PoissonProcess(5.0)
    #     @inferred advance!(pp, 10.0)
        
    #     ipp = InhomogeneousPoissonProcess(t -> 5.0, 10.0)
    #     @inferred advance!(ipp, 10.0)
    # end
end
