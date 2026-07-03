using MonteCarloX
using StatsBase
using Random
using Test

# Adapting the step size during warm-up should drive the acceptance rate to `target`,
# starting from a deliberately far-off step, and still sample the target correctly.
function test_step_size_targets_acceptance()
    pass = true
    rng  = Xoshiro(1)
    alg  = MarkovChainMonteCarlo(rng, x -> -x^2 / 2)     # standard normal, up to a constant

    step = StepSizeAdaptor(8.0; target = 0.4)            # start much too large
    θ    = 0.0
    for _ in 1:20_000                                    # warm-up with adaptation
        θ′  = θ + step_size(step) * randn(rng)
        acc = accept!(alg, θ′, θ)
        acc && (θ = θ′)
        adapt!(step, acc)
    end

    reset!(alg)                                          # measure the frozen sampling phase
    samples = Float64[]
    for _ in 1:100_000
        θ′  = θ + step_size(step) * randn(rng)
        acc = accept!(alg, θ′, θ)
        acc && (θ = θ′)
        push!(samples, θ)
    end

    pass &= check(0.32 < acceptance_rate(alg) < 0.48, "step size: acceptance near target\n")
    pass &= check(isapprox(mean(samples), 0.0; atol = 0.05), "step size: sample mean ≈ 0\n")
    pass &= check(isapprox(std(samples), 1.0; atol = 0.1), "step size: sample std ≈ 1\n")
    return pass
end

# A vector base keeps its ratios fixed while only the overall magnitude adapts.
function test_step_size_vector_keeps_ratios()
    pass = true
    step = StepSizeAdaptor([2.0, 0.5]; target = 0.234)
    for _ in 1:100
        adapt!(step, true)                               # drive the magnitude up
    end
    s = step_size(step)
    pass &= check(s[1] > 2.0 && s[2] > 0.5, "vector step: magnitude grew on acceptance\n")
    pass &= check(isapprox(s[1] / s[2], 4.0), "vector step: ratio preserved\n")
    return pass
end

# reset! restores the initial step and clears the counter.
function test_step_size_reset()
    step = StepSizeAdaptor(1.0)
    adapt!(step, true); adapt!(step, false)
    reset!(step)
    return check(step_size(step) == 1.0, "step size: reset restores base\n")
end

@testset "StepSizeAdaptor" begin
    @testset "targets acceptance rate" begin
        @test test_step_size_targets_acceptance()
    end
    @testset "vector keeps ratios" begin
        @test test_step_size_vector_keeps_ratios()
    end
    @testset "reset" begin
        @test test_step_size_reset()
    end
end
