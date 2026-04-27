@testset "Potentials" begin

    @testset "NoPotential" begin
        pot = NoPotential()
        @test pot(1.0) == 0.0
        @test pot(100.0) == 0.0
        @test cutoff_sq(pot) == Inf
    end

    @testset "NoBondPotential" begin
        pot = NoBondPotential()
        @test pot(1.0) == 0.0
    end

    @testset "NoBendingPotential" begin
        pot = NoBendingPotential()
        @test pot(0.5) == 0.0
    end

    @testset "LennardJonesPotential" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)

        # At r = sigma: V = 4ε(1 - 1) - V_cutoff = -V_cutoff
        r_sq_sigma = 1.0
        # At r² = 1 (r = σ): sixterm = 1/1 = 1, V = 4*(1-1) - v_cutoff
        @test lj(r_sq_sigma) ≈ -lj.v_cutoff

        # At r = 2^(1/6) * sigma (minimum): V = -ε - V_cutoff
        r_min_sq = 2.0^(1.0/3.0)  # (2^(1/6))^2 = 2^(1/3)
        @test lj(r_min_sq) ≈ -1.0 - lj.v_cutoff

        # Beyond cutoff: V = 0
        r_cutoff_sq = lj.r_cutoff_sq
        @test lj(r_cutoff_sq + 0.01) == 0.0

        # Continuity at cutoff: should be ≈ 0
        @test abs(lj(r_cutoff_sq - 1e-10)) < 1e-4

        # Custom parameters
        lj2 = LennardJonesPotential(epsilon=2.0, sigma=1.5, r_cutoff=3.0)
        @test lj2.epsilon == 2.0
        @test lj2.r_cutoff_sq ≈ 9.0
    end

    @testset "FENEPotential" begin
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)

        # At r = 0 (equilibrium for l0=0): V = 0
        @test fene(0.0) ≈ 0.0

        # At r² = 1.0 (r = 1.0, within max extension R=1.5):
        # V = -15 * 2.25 * log(1 - (1/1.5)^2) = -33.75 * log(1 - 4/9)
        r_sq = 1.0
        expected = -0.5 * 30.0 * 1.5^2 * log1p(-1.0/1.5^2)
        @test fene(r_sq) ≈ expected

        # Beyond max extension: V = Inf
        @test fene(1.5^2 + 0.01) == Inf

        # With nonzero l0
        fene2 = FENEPotential(spring_constant=30.0, l0=1.0, l_max=2.0)
        # R = l_max - l0 = 1.0
        @test fene2(1.0) ≈ 0.0  # r = 1 = l0
        @test fene2(4.0 + 0.01) == Inf  # r = 2.0 + tiny > l_max
    end

    @testset "CosineBendingPotential" begin
        bend = CosineBendingPotential(5.0)

        # cos(0) = 1 -> V = 0 (straight chain)
        @test bend(1.0) ≈ 0.0

        # cos(π) = -1 -> V = 2κ = 10
        @test bend(-1.0) ≈ 10.0

        # cos(π/2) = 0 -> V = κ = 5
        @test bend(0.0) ≈ 5.0
    end
end
