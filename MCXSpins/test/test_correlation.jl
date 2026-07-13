using Test
using Random
using MonteCarloX
using MCXSpins

@testset "correlation_length matches second-moment definition" begin
    rng = MersenneTwister(11)
    for dims in ([4, 6], [3, 4, 5])
        sys = VisionConeIsingSystem(dims; κ=0.3)
        init!(sys, :random, rng=rng)
        S0 = float(magnetization(sys))^2
        expected = sum(sqrt(max(S0 / structure_factor(sys, d) - 1, 0.0)) / (2 * sin(π / dims[d]))
                       for d in eachindex(dims)) / length(dims)
        @test correlation_length(sys) ≈ expected
    end
end

@testset "correlation_length of a single defect in an ordered state" begin
    # One flipped spin at r=0: S(k) = 4 exactly per axis, S(0) = (N-2)², so
    # ξ = √((N−2)²/4 − 1) / (2 sin(π/L)) analytically.
    L = 8
    sys = VisionConeIsingSystem([L, L]; κ=0.0)
    init!(sys, :up)
    sys.spins[1] = Int8(-1)
    MCXSpins.recompute_all!(sys)
    N = L * L
    ξ_exact = sqrt((N - 2)^2 / 4 - 1) / (2 * sin(π / L))
    @test correlation_length(sys) ≈ ξ_exact rtol = 1e-12
end

@testset "correlation_length is 0 for uniform configurations" begin
    si = VisionConeIsingSystem([8, 8]; κ=0.5)
    for type in (:up, :down)
        init!(si, type)
        @test correlation_length(si) == 0.0
    end
    sb = VisionConeBlumeCapelSystem([8, 8]; κ=0.5)
    init!(sb, :zero)
    @test correlation_length(sb) == 0.0
end

@testset "correlation_length symmetries and bounds" begin
    rng = MersenneTwister(3)
    sys = VisionConeIsingSystem([6, 6]; κ=0.2)
    init!(sys, :random, rng=rng)
    ξ = correlation_length(sys)
    @test ξ >= 0.0

    # Translation invariance on the periodic lattice
    conf = reshape(copy(sys.spins), geometry(sys))
    sys.spins .= vec(circshift(conf, (2, 1)))
    MCXSpins.recompute_all!(sys)
    @test correlation_length(sys) ≈ ξ

    # Global spin-flip invariance
    sys.spins .= .-sys.spins
    MCXSpins.recompute_all!(sys)
    @test correlation_length(sys) ≈ ξ
end

@testset "correlation_length skips axes with L < 2" begin
    rng = MersenneTwister(5)
    sys = VisionConeIsingSystem([1, 6]; κ=0.0)
    init!(sys, :random, rng=rng)
    S0 = float(magnetization(sys))^2
    expected = sqrt(max(S0 / structure_factor(sys, 2) - 1, 0.0)) / (2 * sin(π / 6))
    @test correlation_length(sys) ≈ expected
end
