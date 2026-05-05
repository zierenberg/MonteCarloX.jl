using SoftMatterSystems
using MonteCarloX
using Test
using Random
using StaticArrays

# ── ParticleGas ──────────────────────────────────────────────────────────────

@testset "ParticleGas" begin
    @testset "Construction" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(10; L=10.0, pair_potential=lj)
        @test gas.N == 10
        @test gas.L ≈ 10.0
        @test length(gas.positions) == 10
    end

    @testset "Density constructor" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(100, 0.1; pair_potential=lj)
        @test gas.L ≈ (100 / 0.1)^(1/3)
    end

    @testset "Init and energy consistency" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(20; L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(123))

        E_cached = energy(gas)
        E_full = energy(gas; full=true)
        @test E_cached ≈ E_full
    end

    @testset "delta_energy correctness" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(10; L=10.0, pair_potential=lj, delta=0.2)
        init!(gas, :random; rng=Xoshiro(456))

        for _ in 1:20
            i = rand(1:10)
            old_pos = gas.positions[i]
            dx = 0.1; dy = 0.2; dz = -0.15
            new_pos = wrap_position(old_pos + SVector{3,Float64}(dx, dy, dz), gas.L)

            E_before = energy(gas; full=true)
            dE = delta_energy(gas, i, old_pos, new_pos)
            E_after = energy(gas; full=true)

            @test dE ≈ E_after - E_before atol=1e-10
        end
    end

    @testset "Metropolis moves maintain energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(30; L=10.0, pair_potential=lj, delta=0.2)
        init!(gas, :random; rng=Xoshiro(789))
        alg = Metropolis(Xoshiro(999); beta=1.0)

        for _ in 1:500
            particle_move!(gas, alg)
        end

        E_cached = energy(gas)
        E_full = energy(gas; full=true)
        @test E_cached ≈ E_full atol=1e-10
        @test 0.0 < acceptance_rate(alg) < 1.0
    end

    @testset "ImportanceSampling moves maintain energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(20; L=10.0, pair_potential=lj, delta=0.2)
        init!(gas, :random; rng=Xoshiro(111))
        alg = Metropolis(Xoshiro(222); beta=0.5)

        for _ in 1:300
            particle_move!(gas, alg)
        end

        E_cached = energy(gas)
        E_full = energy(gas; full=true)
        @test E_cached ≈ E_full atol=1e-10
    end

    @testset "NoPotential gas has zero energy" begin
        gas = ParticleGas(10; L=5.0, pair_potential=NoPotential())
        init!(gas, :random; rng=Xoshiro(42))
        @test energy(gas) ≈ 0.0
    end
end

# ── BeadSpringPolymer ────────────────────────────────────────────────────────

@testset "BeadSpringPolymer" begin
    @testset "Construction" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(2, 5; L=20.0, pair_potential=lj, bond_potential=fene)
        @test poly.M == 2
        @test poly.N == 5
        @test length(poly.positions) == 10
        @test poly.bending_potential isa NoBendingPotential
    end

    @testset "Construction with bending" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(5.0)
        poly = BeadSpringPolymer(2, 5; L=20.0, pair_potential=lj, bond_potential=fene, bending_potential=bend)
        @test poly.bending_potential isa CosineBendingPotential
        @test poly.bending_potential.kappa == 5.0
    end

    @testset "Random walk init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(3, 8; L=20.0, pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(42))

        for pos in poly.positions
            for d in 1:3
                @test 0.0 <= pos[d] < poly.L
            end
        end

        for m in 1:3
            for k in 1:7
                i = (m-1)*8 + k
                j = (m-1)*8 + k + 1
                r_sq = minimum_image_sq(poly.positions[i], poly.positions[j], poly.L)
                @test sqrt(r_sq) ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "Energy decomposition" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(3.0)
        poly = BeadSpringPolymer(2, 10; L=20.0, pair_potential=lj, bond_potential=fene, bending_potential=bend)
        init!(poly, :random_walk; rng=Xoshiro(77))

        E_total = energy(poly)
        E_pair = energy_pair(poly)
        E_bond = energy_bond(poly)
        E_bend = energy_bending(poly)

        @test E_total ≈ E_pair + E_bond + E_bend
        @test isfinite(E_pair)
        @test isfinite(E_bond)
        @test isfinite(E_bend)

        E_full = energy(poly; full=true)
        @test E_total ≈ E_full
    end

    @testset "delta_energy correctness" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(2, 6; L=20.0, pair_potential=lj, bond_potential=fene, delta=0.1)
        init!(poly, :random_walk; rng=Xoshiro(100))

        for _ in 1:20
            idx = rand(1:length(poly.positions))
            old_pos = poly.positions[idx]
            dx = 0.05; dy = -0.03; dz = 0.04
            new_pos = wrap_position(old_pos + SVector{3,Float64}(dx, dy, dz), poly.L)

            E_before = energy(poly; full=true)
            dE = delta_energy(poly, idx, old_pos, new_pos)
            E_after = energy(poly; full=true)

            @test dE ≈ E_after - E_before atol=1e-10
        end
    end

    @testset "Metropolis moves maintain energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(2, 8; L=20.0, pair_potential=lj, bond_potential=fene, delta=0.1)
        init!(poly, :random_walk; rng=Xoshiro(200))
        alg = Metropolis(Xoshiro(300); beta=1.0)

        for _ in 1:500
            monomer_move!(poly, alg)
        end

        E_cached = energy(poly)
        E_full = energy(poly; full=true)
        @test E_cached ≈ E_full atol=1e-10
        @test 0.0 < acceptance_rate(alg) < 1.0
    end

    @testset "Semiflexible polymer Metropolis" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(5.0)
        poly = BeadSpringPolymer(2, 6; L=20.0, pair_potential=lj, bond_potential=fene, bending_potential=bend, delta=0.05)
        init!(poly, :random_walk; rng=Xoshiro(400))
        alg = Metropolis(Xoshiro(500); beta=1.0)

        for _ in 1:500
            monomer_move!(poly, alg)
        end

        E_cached = energy(poly)
        E_full = energy(poly; full=true)
        @test E_cached ≈ E_full atol=1e-10

        # Energy decomposition should still sum correctly
        @test energy(poly) ≈ energy_pair(poly) + energy_bond(poly) + energy_bending(poly) atol=1e-10
    end

    @testset "NoPotential pair with FENE bond" begin
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(1, 4; L=20.0, pair_potential=NoPotential(), bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(55))

        @test energy_pair(poly) ≈ 0.0
    end

    @testset "Single chain (M=1) Metropolis" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(1, 10; L=15.0, pair_potential=lj, bond_potential=fene, delta=0.05)
        init!(poly, :random_walk; rng=Xoshiro(600))
        alg = Metropolis(Xoshiro(700); beta=1.0)

        for _ in 1:200
            monomer_move!(poly, alg)
        end

        E_cached = energy(poly)
        E_full = energy(poly; full=true)
        @test E_cached ≈ E_full atol=1e-10
    end
end
