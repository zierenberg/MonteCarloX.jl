using MonteCarloX: Metropolis, acceptance_rate

"""
    verify_invariants(sys::ParticleSystem)

Check energy consistency. For polymer systems, also checks finite bond lengths.
"""
function verify_invariants(sys::ParticleSystem)
    @assert energy(sys) ≈ energy(sys; full=true) "Cached energy $(energy(sys)) != full $(energy(sys; full=true))"
end

function verify_invariants(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    @assert energy(sys) ≈ energy(sys; full=true) "Cached energy $(energy(sys)) != full $(energy(sys; full=true))"
    # Bond lengths should be finite (not broken by FENE)
    for m in 1:num_polymers(sys)
        mol = sys.molecules[m]
        M = mol.length
        off = mol.offset
        for k in 1:M-1
            r_sq = minimum_image_sq(sys.positions[off+k], sys.positions[off+k+1], sys.L)
            @assert isfinite(mol.bond(r_sq)) "Broken bond in polymer $m between monomers $k and $(k+1)"
        end
    end
end

@testset "MC Updates" begin

    @testset "translate! ParticleGas: per-step invariants" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; N=30, L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(1); β=1.0)
        for _ in 1:500
            translate!(gas, alg, 0.2)
            verify_invariants(gas)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "translate! ParticleGas 2D: per-step invariants" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; D=2, N=20, L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(2); β=1.0)
        for _ in 1:300
            translate!(gas, alg, 0.2)
            verify_invariants(gas)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "translate! BeadSpringPolymer: per-step invariants" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=2, length_poly=8, L=20.0,
            pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(3); β=1.0)
        for _ in 1:500
            translate!(poly, alg, 0.1)
            verify_invariants(poly)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "translate! with bending: per-step invariants" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        bend = CosineBendingPotential(5.0)
        poly = BeadSpringPolymer(; num_poly=2, length_poly=6, L=20.0,
            pair_potential=lj, bond_potential=fene,
            bending_potential=bend)
        init!(poly, :random_walk; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(4); β=1.0)
        for _ in 1:500
            translate!(poly, alg, 0.05)
            verify_invariants(poly)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "translate! chain: per-step invariants" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=2, length_poly=8, L=20.0,
            pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(42))
        alg = Metropolis(Xoshiro(5); β=1.0)
        for _ in 1:500
            translate!(poly, alg, 0.1; chain=true)
            verify_invariants(poly)
        end
        @test acceptance_rate(alg) > 0.0
    end

    @testset "Mixed moves: long run" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        fene = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5)
        poly = BeadSpringPolymer(; num_poly=3, length_poly=8, L=20.0,
            pair_potential=lj, bond_potential=fene)
        init!(poly, :random_walk; rng=Xoshiro(99))
        alg = Metropolis(Xoshiro(0); β=0.5)
        for _ in 1:2000
            translate!(poly, alg, 0.1)
        end
        verify_invariants(poly)
        @test 0.0 < acceptance_rate(alg) < 1.0
    end
end
