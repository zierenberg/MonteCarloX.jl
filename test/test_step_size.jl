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

    step = AdaptiveStep(8.0; target = 0.4)            # start much too large
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
    step = AdaptiveStep([2.0, 0.5]; target = 0.234)
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
    step = AdaptiveStep(1.0)
    adapt!(step, true); adapt!(step, false)
    reset!(step)
    return check(step_size(step) == 1.0, "step size: reset restores base\n")
end

# A two-phase (warm-up then sample) random-walk loop with adaptation recovers a 2-D
# standard normal — and drives the acceptance to the target.
function test_two_phase_adaptation()
    pass = true
    rng  = Xoshiro(1)
    alg  = MarkovChainMonteCarlo(rng, θ -> -0.5 * sum(abs2, θ))
    step = AdaptiveStep(3.0; target = 0.234)
    θ    = [0.0, 0.0]

    for _ in 1:5_000                                     # warm-up: adapt the step
        θ′       = θ .+ step_size(step) .* randn(rng, 2)
        accepted = accept!(alg, θ′, θ)
        accepted && (θ = θ′)
        adapt!(step, accepted)
    end
    reset!(alg)
    Δ = step_size(step)
    S = zeros(2, 40_000)
    for i in 1:40_000                                    # sampling
        θ′ = θ .+ Δ .* randn(rng, 2)
        accept!(alg, θ′, θ) && (θ = θ′)
        S[:, i] = θ
    end

    pass &= check(0.15 < acceptance_rate(alg) < 0.35, "two-phase: acceptance near target\n")
    pass &= check(isapprox(vec(mean(S, dims = 2)), [0.0, 0.0]; atol = 0.05), "two-phase: means ≈ 0\n")
    pass &= check(all(isapprox.(vec(std(S, dims = 2)), 1.0; atol = 0.1)), "two-phase: stds ≈ 1\n")
    return pass
end

@testset "AdaptiveStep" begin
    @testset "targets acceptance rate" begin
        @test test_step_size_targets_acceptance()
    end
    @testset "vector keeps ratios" begin
        @test test_step_size_vector_keeps_ratios()
    end
    @testset "reset" begin
        @test test_step_size_reset()
    end
    @testset "two-phase adaptation" begin
        @test test_two_phase_adaptation()
    end
end
