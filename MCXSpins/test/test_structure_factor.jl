using Test
using Random
using MonteCarloX
using MCXSpins

# Independent reference: explicit coordinates via CartesianIndices and complex exponentials,
# cross-checking the stride-based coordinate decoding in structure_factor.
function reference_structure_factor(spins, dims, d)
    k = 2π / dims[d]
    lin = LinearIndices(dims)
    z = sum(Int(spins[lin[I]]) * cis(k * (I[d] - 1)) for I in CartesianIndices(dims))
    return abs2(z)
end

@testset "structure_factor matches Fourier reference" begin
    rng = MersenneTwister(42)
    for (make, dims) in ((d -> VisionConeIsingSystem(d; κ=0.3), [4, 6]),
                         (d -> VisionConeIsingSystem(d; κ=0.3), [3, 4, 5]),
                         (d -> VisionConeBlumeCapelSystem(d; κ=0.3, D=0.5), [6, 4]))
        sys = make(dims)
        init!(sys, :random, rng=rng)
        for d in eachindex(dims)
            ref = reference_structure_factor(sys.spins, geometry(sys), d)
            @test isapprox(structure_factor(sys, d), ref; atol=1e-8, rtol=1e-10)
        end
    end
end

@testset "structure_factor of a single localized spin is 1" begin
    # |σ e^{ikr}|² = 1 for a single unit spin, independent of its position and the axis.
    sys = VisionConeBlumeCapelSystem([4, 6]; κ=0.0, D=0.0)
    for site in (1, 7, 24)
        init!(sys, :zero)
        sys.spins[site] = Int8(1)
        MCXSpins.recompute_all!(sys)
        @test structure_factor(sys, 1) ≈ 1.0
        @test structure_factor(sys, 2) ≈ 1.0
    end
end

@testset "structure_factor vanishes for uniform configurations" begin
    sys = VisionConeIsingSystem([8, 8]; κ=0.5)
    for type in (:up, :down)
        init!(sys, type)
        @test structure_factor(sys, 1) ≈ 0.0 atol = 1e-20
        @test structure_factor(sys, 2) ≈ 0.0 atol = 1e-20
    end
end

@testset "structure_factor symmetries" begin
    rng = MersenneTwister(7)
    sys = VisionConeIsingSystem([4, 6]; κ=0.3)
    init!(sys, :random, rng=rng)
    S = [structure_factor(sys, d) for d in 1:2]

    # Global spin flip: |Σ(-σ)e^{ikr}|² = |Σσe^{ikr}|²
    sys.spins .= .-sys.spins
    MCXSpins.recompute_all!(sys)
    @test [structure_factor(sys, d) for d in 1:2] ≈ S

    # Translation on the periodic lattice only adds a phase to the Fourier sum
    conf = reshape(copy(sys.spins), geometry(sys))
    sys.spins .= vec(circshift(conf, (1, 2)))
    MCXSpins.recompute_all!(sys)
    @test [structure_factor(sys, d) for d in 1:2] ≈ S
end

@testset "structure_factor is 0 along axes with L < 2" begin
    rng = MersenneTwister(5)
    sys = VisionConeIsingSystem([1, 6]; κ=0.0)
    init!(sys, :random, rng=rng)
    @test structure_factor(sys, 1) == 0.0
    @test structure_factor(sys, 2) > 0.0
end
