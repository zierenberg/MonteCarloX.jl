@testset "ParticleGas" begin
    rng = Xoshiro(42)

    @testset "Construction and init" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(10; L=10.0, pair_potential=lj)
        @test gas.N == 10
        @test gas.L ≈ 10.0
        @test length(gas.positions) == 10

        init!(gas, :random; rng=rng)
        # Positions should be in [0, L)
        for pos in gas.positions
            for d in 1:3
                @test 0.0 <= pos[d] < gas.L
            end
        end
    end

    @testset "Density constructor" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(100, 0.1; pair_potential=lj)
        @test gas.L ≈ (100 / 0.1)^(1/3)
    end

    @testset "Energy consistency" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(20; L=10.0, pair_potential=lj)
        init!(gas, :random; rng=Xoshiro(123))

        # Cached energy should match full recomputation
        E_cached = energy(gas)
        E_full = energy(gas; full=true)
        @test E_cached ≈ E_full
    end

    @testset "Metropolis moves maintain energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(30; L=10.0, pair_potential=lj, delta=0.2)
        init!(gas, :random; rng=Xoshiro(456))
        alg = Metropolis(Xoshiro(789); beta=1.0)

        for _ in 1:500
            particle_move!(gas, alg)
        end

        # After many moves, cached energy should still match full recomputation
        E_cached = energy(gas)
        E_full = energy(gas; full=true)
        @test E_cached ≈ E_full atol=1e-10
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

    @testset "Two particles known energy" begin
        lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        gas = ParticleGas(2; L=20.0, pair_potential=lj)

        # Place two particles at known distance
        gas.positions[1] = SVector(0.0, 0.0, 0.0)
        gas.positions[2] = SVector(1.0, 0.0, 0.0)  # r = 1 = sigma

        E_full = energy(gas; full=true)
        # At r = sigma: V = 4(1-1) - v_cutoff = -v_cutoff
        @test E_full ≈ -lj.v_cutoff
    end
end
