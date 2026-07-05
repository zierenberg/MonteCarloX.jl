using MonteCarloX
using StatsBase
using Random
using Test

# ESS ≈ N for an uncorrelated chain, and far below N for a strongly correlated one.
function test_ess()
    pass = true
    rng = Xoshiro(1)

    iid = randn(rng, 10_000)
    pass &= check(ess(iid) > 6_000, "ess: iid close to N\n")

    ar = zeros(10_000); ar[1] = randn(rng)          # AR(1), φ=0.8 → τ_int≈4.5, ESS≈N/9
    for t in 2:10_000
        ar[t] = 0.8 * ar[t-1] + randn(rng)
    end
    pass &= check(ess(ar) < 3_000, "ess: correlated well below N\n")
    pass &= check(ess(ar) < ess(iid), "ess: correlated < iid\n")
    return pass
end

# R̂ ≈ 1 for chains from the same distribution, > 1 when they disagree.
function test_rhat()
    pass = true
    rng = Xoshiro(1)

    same = [randn(rng, 2_000) for _ in 1:4]
    pass &= check(rhat(same) < 1.05, "rhat: converged ≈ 1\n")

    apart = [randn(rng, 2_000) .+ 5.0 * (m - 1) for m in 1:4]   # different means
    pass &= check(rhat(apart) > 1.5, "rhat: unconverged > 1\n")
    return pass
end

@testset "Diagnostics" begin
    @testset "ess (autocorrelation)" begin
        @test test_ess()
    end
    @testset "rhat (Gelman-Rubin)" begin
        @test test_rhat()
    end
end
