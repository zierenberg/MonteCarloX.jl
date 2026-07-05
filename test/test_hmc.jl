using MonteCarloX
using LogDensityProblems
using LogDensityProblemsAD
using ForwardDiff
import LinearAlgebra              # `import` (not `using`): its `rank`/`size` exports would
                                  # clash with MonteCarloX's in the shared test `Main`
using StatsBase
using Random
using Test

# Educational: a minimal fixed-path Hamiltonian Monte Carlo, driven by the gradient
# that the LogDensityProblems bridge exposes for a MonteCarloX `LogDensityTarget`.
# It both shows *how* HMC uses ∇log p (the leapfrog integrator + a Metropolis accept on
# the Hamiltonian) and verifies the bridge end to end: LogDensityTarget → AD gradient.
#
# HMC augments the parameters θ with a momentum p ~ N(0, I) and treats
#   H(θ, p) = -logdensity(θ) + ½‖p‖²
# as an energy. Leapfrog integration follows the level sets of H, so a long proposal
# stays high-probability; the accept step corrects the integrator's small error.

# One HMC transition: L leapfrog steps of size ϵ, then Metropolis on H.
function hmc_step(θ, ℓ, ϵ, L, rng)
    ∇(x) = LogDensityProblems.logdensity_and_gradient(ℓ, x)   # (logp, gradient) via the bridge
    p      = randn(rng, length(θ))
    lp0, g = ∇(θ)
    H0     = -lp0 + 0.5 * sum(abs2, p)
    θc     = copy(θ)
    for _ in 1:L
        p    = p .+ 0.5 * ϵ .* g          # half momentum kick
        θc   = θc .+ ϵ .* p               # full position drift
        _, g = ∇(θc)
        p    = p .+ 0.5 * ϵ .* g          # half momentum kick
    end
    lp, _ = ∇(θc)
    H     = -lp + 0.5 * sum(abs2, p)
    return rand(rng) < exp(H0 - H) ? (θc, true) : (θ, false)
end

# HMC recovers a correlated 2-D Gaussian target expressed as a MonteCarloX target.
function test_hmc_recovers_gaussian()
    pass = true
    Σinv   = [2.0 0.8; 0.8 1.0]
    target = LogDensityTarget(θ -> -0.5 * LinearAlgebra.dot(θ, Σinv, θ), 2)   # MCX target
    ℓ      = ADgradient(:ForwardDiff, target)                                  # bridge → AD gradient

    rng = Xoshiro(1)
    θ   = [0.0, 0.0]
    S   = zeros(2, 20_000)
    acc = 0
    for i in 1:20_000
        θ, a   = hmc_step(θ, ℓ, 0.3, 8, rng)
        acc   += a
        S[:, i] = θ
    end

    Σ = LinearAlgebra.inv(Σinv)
    pass &= check(acc / 20_000 > 0.6, "hmc: healthy acceptance\n")
    pass &= check(isapprox(vec(mean(S, dims = 2)), [0.0, 0.0]; atol = 0.1), "hmc: means ≈ 0\n")
    pass &= check(isapprox(cov(S; dims = 2), Σ; atol = 0.15), "hmc: covariance ≈ Σ\n")
    return pass
end

@testset "HMC via LogDensityProblems bridge" begin
    @test test_hmc_recovers_gaussian()
end
