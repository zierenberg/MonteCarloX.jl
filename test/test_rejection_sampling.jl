using MonteCarloX
using Distributions
using Random
using StatsBase
using Test

# envelope == target: log_ratio == 0 everywhere ⇒ accept with prob 1.
function test_rejection_always_accept()
    pass = true
    rng = MersenneTwister(1)
    alg = RejectionSampling(rng, x -> 0.0, x -> 0.0)
    for _ in 1:1000
        accept!(alg, rand(rng))
    end
    pass &= check(alg.steps == 1000, "always-accept: steps counted\n")
    pass &= check(alg.accepted == 1000, "always-accept: all accepted\n")
    pass &= check(acceptance_rate(alg) == 1.0, "always-accept: rate == 1\n")
    return pass
end

function test_rejection_reset()
    pass = true
    rng = MersenneTwister(1)
    alg = RejectionSampling(rng, x -> 0.0, x -> 0.0)
    for _ in 1:10
        accept!(alg, rand(rng))
    end
    reset!(alg)
    pass &= check(alg.steps == 0 && alg.accepted == 0, "reset: counters zero\n")
    return pass
end

# Draw exact Normal(0,1) samples through a broader Normal(0,2) proposal.
# The envelope g = M·q has logg = logq + logM with logM = sup(logp - logq);
# acceptance ≈ Z_p/Z_g = 1/M since both p and q are normalized.
function test_rejection_density()
    pass = true
    rng = MersenneTwister(20)
    q, p = Normal(0.0, 2.0), Normal(0.0, 1.0)
    logM = logpdf(p, 0.0) - logpdf(q, 0.0)          # ratio peaks at x = 0
    alg = RejectionSampling(rng, x -> logpdf(q, x) + logM, p)

    samples = Float64[]
    while length(samples) < 50_000
        x = rand(rng, q)
        accept!(alg, x) && push!(samples, x)
    end

    pass &= check(isapprox(mean(samples), 0.0; atol = 0.02), "density: mean ≈ 0\n")
    pass &= check(isapprox(std(samples), 1.0; atol = 0.02), "density: std ≈ 1\n")
    pass &= check(isapprox(acceptance_rate(alg), exp(-logM); atol = 0.02), "density: acceptance ≈ 1/M\n")
    return pass
end

# Hit-or-miss integration of ∫₀¹ sin(x) dx. Sample (x,y) uniformly in the unit box
# and accept when y < f(x); the accepted points are uniform under the curve, and
# the acceptance fraction equals the integral. In rejection terms: uniform box
# envelope (logg = 0), target uniform on the region under the curve (logp = 0
# inside, -Inf outside) — so acceptance_rate estimates Z_target/Z_envelope.
function test_rejection_integration()
    pass = true
    rng = MersenneTwister(7)
    f(x) = sin(x)
    under_curve((x, y)) = (0.0 ≤ y ≤ f(x)) ? 0.0 : -Inf
    alg = RejectionSampling(rng, _ -> 0.0, under_curve)
    for _ in 1:1_000_000
        accept!(alg, (rand(rng), rand(rng)))
    end
    pass &= check(isapprox(acceptance_rate(alg), 1 - cos(1); atol = 0.003), "integration: acceptance ≈ ∫₀¹ sin\n")
    return pass
end

@testset "RejectionSampling" begin
    @testset "always accept" begin
        @test test_rejection_always_accept()
    end
    @testset "reset counters" begin
        @test test_rejection_reset()
    end
    @testset "envelope violation throws" begin
        # target (logp ≡ 0) exceeds envelope (logg ≡ -1) ⇒ violation.
        alg = RejectionSampling(MersenneTwister(1), x -> -1.0, x -> 0.0)
        @test_throws ArgumentError accept!(alg, 0.5)
    end
    @testset "density sampling" begin
        @test test_rejection_density()
    end
    @testset "hit-or-miss integration" begin
        @test test_rejection_integration()
    end
end
