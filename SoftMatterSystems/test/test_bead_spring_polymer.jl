@testset "BeadSpringPolymer" begin

    @testset "Construction" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(2, 5; L=20.0,
            pair_potential=lj, bond_potential=fene)
        @test poly.M == 2
        @test poly.N == 5
        @test length(poly.positions) == 10
        @test poly.bending_potential isa NoBendingPotential
    end

    @testset "Construction with bending" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(5.0)
        poly = BeadSpringPolymer(2, 5; L=20.0,
            pair_potential=lj, bond_potential=fene,
            bending_potential=bend)
        @test poly.bending_potential isa CosineBendingPotential
        @test poly.bending_potential.kappa == 5.0
    end

    @testset "Random walk init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(3, 8; L=20.0,
            pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(42))

        # All positions in [0, L)
        for pos in poly.positions
            for d in 1:3
                @test 0.0 <= pos[d] < poly.L
            end
        end

        # Bond lengths should be ≈ 1 (random walk step size)
        for m in 1:3
            for k in 1:7
                i = (m-1)*8 + k
                j = (m-1)*8 + k + 1
                r_sq = minimum_image_sq(poly.positions[i], poly.positions[j], poly.L)
                @test sqrt(r_sq) ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "Energy consistency after init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(3.0)
        poly = BeadSpringPolymer(2, 10; L=20.0,
            pair_potential=lj, bond_potential=fene,
            bending_potential=bend)
        init!(poly, :random_walk; rng=Xoshiro(77))

        E_total = energy(poly)
        E_pair = energy_pair(poly)
        E_bond = energy_bond(poly)
        E_bend = energy_bending(poly)

        @test E_total ≈ E_pair + E_bond + E_bend

        # Full recomputation should match
        E_full = energy(poly; full=true)
        @test E_total ≈ E_full
    end

    @testset "Metropolis moves maintain energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(2, 8; L=20.0,
            pair_potential=lj, bond_potential=fene, delta=0.1)
        init!(poly, :random_walk; rng=Xoshiro(100))
        alg = Metropolis(Xoshiro(200); beta=1.0)

        for _ in 1:500
            monomer_move!(poly, alg)
        end

        E_cached = energy(poly)
        E_full = energy(poly; full=true)
        @test E_cached ≈ E_full atol=1e-10
    end

    @testset "Semiflexible polymer Metropolis" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(5.0)
        poly = BeadSpringPolymer(2, 6; L=20.0,
            pair_potential=lj, bond_potential=fene,
            bending_potential=bend, delta=0.05)
        init!(poly, :random_walk; rng=Xoshiro(300))
        alg = Metropolis(Xoshiro(400); beta=1.0)

        for _ in 1:500
            monomer_move!(poly, alg)
        end

        E_cached = energy(poly)
        E_full = energy(poly; full=true)
        @test E_cached ≈ E_full atol=1e-10
    end

    @testset "NoPotential pair with FENE bond" begin
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(1, 4; L=20.0,
            pair_potential=NoPotential(), bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(55))

        @test energy_pair(poly) ≈ 0.0
        @test energy_bond(poly) > 0.0  # FENE at r=1 is positive
    end

    @testset "Energy decomposition" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(2.0)
        poly = BeadSpringPolymer(2, 5; L=20.0,
            pair_potential=lj, bond_potential=fene,
            bending_potential=bend)
        init!(poly, :random_walk; rng=Xoshiro(88))

        # Each component should be finite
        @test isfinite(energy_pair(poly))
        @test isfinite(energy_bond(poly))
        @test isfinite(energy_bending(poly))

        # Sum should equal total
        @test energy(poly) ≈ energy_pair(poly) + energy_bond(poly) + energy_bending(poly)
    end
end
