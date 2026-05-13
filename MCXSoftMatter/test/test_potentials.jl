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

        # At r = sigma: V = 4(1-1) - v_cutoff = -v_cutoff
        @test lj(1.0) ≈ -lj.v_cutoff

        # At r = 2^(1/6)*sigma (minimum): V = -epsilon - v_cutoff
        r_min_sq = 2.0^(1.0/3.0)
        @test lj(r_min_sq) ≈ -1.0 - lj.v_cutoff

        # Beyond cutoff: V = 0
        @test lj(lj.r_cutoff_sq + 0.01) == 0.0

        # Continuity at cutoff
        @test abs(lj(lj.r_cutoff_sq - 1e-10)) < 1e-4

        # Custom parameters
        lj2 = LennardJonesPotential(epsilon=2.0, sigma=1.5, r_cutoff=3.0)
        @test lj2.epsilon == 2.0
        @test lj2.r_cutoff_sq ≈ 9.0
    end

    @testset "FENEPotential" begin
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)

        # At r = 0: V = 0
        @test fene(0.0) ≈ 0.0

        # At r^2 = 1.0 (within max extension R=1.5)
        expected = -0.5 * 30.0 * 1.5^2 * log1p(-1.0/1.5^2)
        @test fene(1.0) ≈ expected

        # Beyond max extension: V = Inf
        @test fene(1.5^2 + 0.01) == Inf

        # With nonzero l0
        fene2 = FENEPotential(spring_constant=30.0, l0=1.0, l_max=2.0)
        @test fene2(1.0) ≈ 0.0  # r = 1 = l0
        @test fene2(4.0 + 0.01) == Inf  # r > l_max
    end

    @testset "CosineBendingPotential" begin
        bend = CosineBendingPotential(5.0)
        @test bend(1.0) ≈ 0.0    # cos(0) = 1 -> straight
        @test bend(-1.0) ≈ 10.0  # cos(pi) = -1 -> hairpin
        @test bend(0.0) ≈ 5.0    # cos(pi/2) = 0
    end
end
