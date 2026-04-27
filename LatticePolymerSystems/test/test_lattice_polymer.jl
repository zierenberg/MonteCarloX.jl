using LatticePolymerSystems
using MonteCarloX
using Test
using Random

@testset "LatticePolymer" begin
    @testset "Construction and ordered init" begin
        sys = LatticePolymer(10; M=4, N=5, J_intra=0.0, J_inter=1.0)
        init!(sys, :ordered)
        @test length(sys.polymers) == 4
        for m in 1:4
            @test length(sys.polymers[m]) == 5
        end
        # All sites occupied by polymers should have correct occupant
        for m in 1:4
            for site in sys.polymers[m]
                @test sys.site_occupant[site] == m
            end
        end
        # Total occupied sites
        @test count(>(0), sys.site_occupant) == 20
    end

    @testset "Random init" begin
        sys = LatticePolymer(10; M=8, N=6, J_intra=0.0, J_inter=1.0)
        init!(sys, :random; rng=Xoshiro(42))
        for m in 1:8
            @test length(sys.polymers[m]) == 6
            # Check self-avoidance: all monomer sites unique
            @test length(unique(sys.polymers[m])) == 6
            # Check connectivity: consecutive monomers are lattice neighbors
            for k in 1:5
                site_k = sys.polymers[m][k]
                site_k1 = sys.polymers[m][k+1]
                @test site_k1 ∈ sys.nbrs[site_k]
            end
        end
        @test count(>(0), sys.site_occupant) == 48
    end

    @testset "Cached energy matches full recomputation" begin
        sys = LatticePolymer(8; M=4, N=6, J_intra=1.0, J_inter=1.5)
        init!(sys, :random; rng=Xoshiro(99))
        E_cached = energy(sys)
        E_full = energy(sys; full=true)
        @test E_cached ≈ E_full
    end

    @testset "Metropolis integration: energy stays consistent" begin
        rng = Xoshiro(55)
        sys = LatticePolymer(10; M=4, N=8, J_intra=0.5, J_inter=1.0)
        init!(sys, :random; rng=rng)
        alg = Metropolis(rng; β=0.3)

        for _ in 1:5000
            polymer_move!(sys, alg)
        end

        # Cached energy must match full recomputation
        E_cached = energy(sys)
        E_full = energy(sys; full=true)
        @test E_cached ≈ E_full

        # Self-avoidance maintained
        occupied = count(>(0), sys.site_occupant)
        @test occupied == 4 * 8

        # Connectivity maintained
        for m in 1:4
            @test length(sys.polymers[m]) == 8
            for k in 1:7
                @test sys.polymers[m][k+1] ∈ sys.nbrs[sys.polymers[m][k]]
            end
        end

        @test 0.0 < acceptance_rate(alg) < 1.0
    end

    @testset "Single-monomer polymer behaves like lattice gas" begin
        # With N=1, only translation moves are used
        rng = Xoshiro(77)
        sys = LatticePolymer(6; M=10, N=1, J_intra=0.0, J_inter=1.0)
        init!(sys, :random; rng=rng)
        alg = Metropolis(rng; β=0.5)

        for _ in 1:2000
            polymer_move!(sys, alg)
        end

        E_cached = energy(sys)
        E_full = energy(sys; full=true)
        @test E_cached ≈ E_full
        @test count(>(0), sys.site_occupant) == 10
    end

    @testset "Known configuration: isolated polymer" begin
        sys = LatticePolymer(10; M=1, N=3, J_intra=1.0, J_inter=0.0)
        init!(sys, :ordered)
        # A straight rod of 3 monomers has 2 backbone bonds and 0 extra intra contacts
        @test sys.num_intra_contacts == 0
        @test sys.num_inter_contacts == 0
        @test energy(sys) ≈ 0.0
    end

    @testset "Polymer observables" begin
        sys = LatticePolymer(20; M=2, N=6, J_intra=0.0, J_inter=1.0)
        init!(sys, :ordered)

        # Radius of gyration should be positive for non-trivial polymer
        rg2 = radius_of_gyration_sq(sys, 1)
        @test rg2 > 0.0

        # Center of mass should be in bounds
        cm = center_of_mass(sys, 1)
        @test 0.0 <= cm[1] < 20.0
        @test 0.0 <= cm[2] < 20.0
        @test 0.0 <= cm[3] < 20.0

        # End-to-end distance for a straight rod of N=6 along z: should be (N-1)^2 = 25
        r2 = end_to_end_distance_sq(sys, 1)
        @test r2 ≈ 25.0
    end
end
