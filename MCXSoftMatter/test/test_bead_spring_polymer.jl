using StaticArrays: SVector

@testset "BeadSpringPolymer" begin

    @testset "Construction" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=2, length_poly=5, L=20.0,
            pair_potential=lj, bond_potential=fene)
        @test num_polymers(poly) == 2
        @test polymer_length(poly) == 5
        @test length(poly.positions) == 10
        @test poly.molecules[1].bend isa NoBendingPotential

        # With bending
        bend = CosineBendingPotential(5.0)
        poly2 = BeadSpringPolymer(; num_poly=2, length_poly=5, L=20.0,
            pair_potential=lj, bond_potential=fene, bending_potential=bend)
        @test poly2.molecules[1].bend.kappa == 5.0
    end

    @testset "Random walk init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=3, length_poly=8, L=20.0,
            pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(42))

        # Positions in box
        @test all(pos -> all(d -> 0.0 <= pos[d] < poly.env.L[d], 1:3), poly.positions)

        # Bond lengths = 1.0 (random walk step size)
        for m in 1:3, k in 1:7
            i = (m-1)*8 + k
            j = (m-1)*8 + k + 1
            r_sq = distance_sq(poly.env, poly.positions[i], poly.positions[j])
            @test sqrt(r_sq) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(3.0)
        poly = BeadSpringPolymer(; num_poly=2, length_poly=10, L=20.0,
            pair_potential=lj, bond_potential=fene, bending_potential=bend)
        init!(poly, :random_walk; rng=Xoshiro(77))

        # Cached matches full recompute
        @test energy(poly) ≈ energy(poly; full=true)
        # Decomposition sums to total
        @test energy(poly) ≈ energy_pair(poly) + energy_bond(poly) + energy_bending(poly)

        # NoPotential pair → zero pair energy
        poly0 = BeadSpringPolymer(; num_poly=1, length_poly=4, L=20.0,
            pair_potential=NoPotential(), bond_potential=fene)
        init!(poly0, :random_walk; rng=Xoshiro(55))
        @test energy_pair(poly0) ≈ 0.0
        @test energy_bond(poly0) > 0.0
    end

    @testset "Observables: straight rod" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=1, length_poly=5, L=20.0,
            pair_potential=lj, bond_potential=fene)
        for k in 1:5
            poly.positions[k] = SVector(Float64(k-1), 0.0, 0.0)
        end

        # End-to-end: |4 - 0|^2 = 16
        @test end_to_end_distance_sq(poly, 1) ≈ 16.0

        # CM: mean(0:4) = 2.0 in x
        cm = center_of_mass(poly, 1)
        @test cm[1] ≈ 2.0
        @test cm[2] ≈ 0.0
        @test cm[3] ≈ 0.0

        # Rg^2: var({0,1,2,3,4}) = 2.0
        @test radius_of_gyration_sq(poly, 1) ≈ 2.0

        # Gyration tensor: all variance in x
        G = gyration_tensor(poly, 1)
        @test sum(G[d,d] for d in 1:3) ≈ radius_of_gyration_sq(poly, 1)
        @test G[1,1] ≈ 2.0
        @test G[2,2] ≈ 0.0 atol=1e-12
    end

    @testset "Heterogeneous lengths" begin
        lj   = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=3, lengths=[4, 6, 5], L=20.0,
            pair_potential=lj, bond_potential=fene)
        @test num_polymers(poly) == 3
        @test polymer_length(poly, 1) == 4
        @test polymer_length(poly, 2) == 6
        @test polymer_length(poly, 3) == 5
        @test num_particles(poly) == 15
        @test [m.offset for m in poly.molecules] == [0, 4, 10]
        init!(poly, :random_walk; rng=Xoshiro(99))
        @test energy(poly) ≈ energy(poly; full=true)
    end
end
