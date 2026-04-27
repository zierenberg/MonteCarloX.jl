using LatticePolymerSystems
using MonteCarloX
using Test
using Random

@testset "LatticeGas" begin
    @testset "Construction and init" begin
        sys = LatticeGas(4; J=1.0, N_particles=10)
        init!(sys, :random; rng=Xoshiro(42))
        @test sum(sys.occupation) == 10
        @test length(sys.occupied_sites) == 10
        @test length(sys.empty_sites) == 4^3 - 10
    end

    @testset "Ordered init" begin
        sys = LatticeGas(4; J=1.0, N_particles=8)
        init!(sys, :ordered)
        @test all(sys.occupation[1:8])
        @test !any(sys.occupation[9:end])
    end

    @testset "Index array consistency" begin
        sys = LatticeGas(5; J=1.0, N_particles=30)
        init!(sys, :random; rng=Xoshiro(123))
        for site in 1:sys.N_sites
            if sys.occupation[site]
                idx = sys.site_to_occupied_idx[site]
                @test idx > 0
                @test sys.occupied_sites[idx] == site
                @test sys.site_to_empty_idx[site] == 0
            else
                idx = sys.site_to_empty_idx[site]
                @test idx > 0
                @test sys.empty_sites[idx] == site
                @test sys.site_to_occupied_idx[site] == 0
            end
        end
    end

    @testset "Cached energy matches full recomputation" begin
        sys = LatticeGas(5; J=1.5, N_particles=30)
        init!(sys, :random; rng=Xoshiro(99))
        E_cached = energy(sys)
        E_full = energy(sys; full=true)
        @test E_cached == E_full
    end

    @testset "Delta energy correctness" begin
        rng = Xoshiro(77)
        sys = LatticeGas(5; J=1.0, N_particles=30)
        init!(sys, :random; rng=rng)

        for _ in 1:100
            occ_idx = rand(rng, 1:sys.N_particles)
            emp_idx = rand(rng, 1:(sys.N_sites - sys.N_particles))
            occ_site = sys.occupied_sites[occ_idx]
            emp_site = sys.empty_sites[emp_idx]

            E_before = energy(sys; full=true)
            ΔE = delta_energy(sys, occ_site, emp_site)

            # Manually perform swap
            sys.occupation[occ_site] = false
            sys.occupation[emp_site] = true
            E_after = energy(sys; full=true)
            # Undo
            sys.occupation[occ_site] = true
            sys.occupation[emp_site] = false

            @test ΔE ≈ E_after - E_before
        end
    end

    @testset "Metropolis integration: energy stays consistent" begin
        rng = Xoshiro(55)
        sys = LatticeGas(6; J=1.0, N_particles=40)
        init!(sys, :random; rng=rng)
        alg = Metropolis(rng; β=0.5)

        for _ in 1:5000
            kawasaki_move!(sys, alg)
        end

        # Cached energy must match full recomputation
        E_cached = energy(sys)
        E_full = energy(sys; full=true)
        @test E_cached ≈ E_full

        # Particle number conserved
        @test sum(sys.occupation) == 40
        @test length(sys.occupied_sites) == 40
        @test length(sys.empty_sites) == 6^3 - 40

        # Acceptance rate should be between 0 and 1
        @test 0.0 < acceptance_rate(alg) < 1.0
    end

    @testset "Known configuration: no contacts" begin
        # L=4, place one particle -> 0 contacts if no neighbors occupied
        sys = LatticeGas(4; J=1.0, N_particles=1)
        init!(sys, :ordered)
        # Site 1 = (0,0,0), no other particles -> 0 contacts
        @test energy(sys) == 0.0
        @test sys.num_contacts == 0
    end

    @testset "Known configuration: two adjacent particles" begin
        sys = LatticeGas(4; J=1.0, N_particles=2)
        # Place at sites 1 and 2 (which are (0,0,0) and (1,0,0) -> neighbors)
        init!(sys, :ordered)
        @test sys.num_contacts == 1
        @test energy(sys) == -1.0
    end
end
