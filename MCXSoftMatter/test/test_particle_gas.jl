using StaticArrays: SVector

@testset "ParticleGas" begin

    @testset "Construction and init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; N=20, L=10.0, pair_potential=lj)
        @test num_particles(gas) == 20
        @test all(gas.box.L .≈ 10.0)

        init!(gas, :random; rng=Xoshiro(42))
        for pos in gas.positions
            @test all(d -> 0.0 <= pos[d] < gas.box.L[d], 1:3)
        end
    end

    @testset "Density constructor" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; N=100, rho=0.1, pair_potential=lj)
        @test all(gas.box.L .≈ (100 / 0.1)^(1/3))
    end

    @testset "2D" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; D=2, N=10, L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(42))
        @test length(gas.positions[1]) == 2
        @test all(pos -> all(d -> 0.0 <= pos[d] < gas.box.L[d], 1:2), gas.positions)
    end

    @testset "Energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(; N=20, L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(123))
        @test energy(gas) ≈ energy(gas; full=true)

        # NoPotential → zero energy
        gas0 = ParticleGas(; N=10, L=5.0, pair_potential=NoPotential())
        init!(gas0, :random; rng=Xoshiro(42))
        @test energy(gas0) ≈ 0.0

        # Two particles at r=sigma: V = 4(1-1) - v_cutoff = -v_cutoff
        gas2 = ParticleGas(; N=2, L=20.0, pair_potential=lj)
        gas2.positions[1] = SVector(0.0, 0.0, 0.0)
        gas2.positions[2] = SVector(1.0, 0.0, 0.0)
        @test energy(gas2; full=true) ≈ -lj.v_cutoff
    end
end
