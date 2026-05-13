using MCXLatticeMatter
using Test
using Random

@testset "LatticePolymer" begin
    # ── Construction & Init ──────────────────────────────────────────────────
    @testset "Ordered init" begin
        sys = LatticePolymer(; dims=[10, 10], num_poly=4, length_poly=5)
        init!(sys, :ordered)
        @test num_polymers(sys) == 4
        @test all(n -> polymer_length(sys, n) == 5, 1:4)
        @test count(>(0), sys.state) == 20
    end

    @testset "Heterogeneous lengths" begin
        sys = LatticePolymer(; dims=[10, 10], polys=[3, 5, 4])
        init!(sys, :ordered)
        @test [polymer_length(sys, n) for n in 1:3] == [3, 5, 4]
        @test num_polymers(sys) == 3
    end

    @testset "Random init: self-avoidance and connectivity" begin
        sys = LatticePolymer(; dims=[10, 10], num_poly=8, length_poly=6)
        init!(sys, :random; rng=Xoshiro(42))
        for n in 1:8
            # test that all polymers have 6 unique sites (self-avoidance)
            @test length(unique(sys.polymers[n])) == 6
            # test that consecutive monomers are neighbors (connectivity)
            for m in 1:5
                s1 = coords_to_site(sys.polymers[n][m], sys.dims)
                s2 = coords_to_site(sys.polymers[n][m+1], sys.dims)
                @test s2 ∈ sys.neighbors[s1]
            end
        end
    end

    # ── Energy ───────────────────────────────────────────────────────────────
    @testset "Energy" begin
        sys = LatticePolymer(; dims=[8, 8], num_poly=4, length_poly=6, J_intra=1.0, J_inter=1.5)
        init!(sys, :random; rng=Xoshiro(99))
        @test energy(sys) ≈ energy(sys; full=true)

        # Isolated 3-mer: 2 backbone bonds → E = -2 J_intra
        sys2 = LatticePolymer(; dims=[10, 10], num_poly=1, length_poly=3, J_intra=1.0, J_inter=0.0)
        init!(sys2, :ordered)
        @test energy(sys2) ≈ -2.0

        # Single monomer: no contacts
        sys3 = LatticePolymer(; dims=[10, 10], num_poly=1, length_poly=1, J_intra=1.0, J_inter=1.0)
        init!(sys3, :ordered)
        @test energy(sys3) ≈ 0.0
    end

    # ── Observables on known geometry ────────────────────────────────────────
    # Ordered init: straight rod along y at x=0, monomers at (0,0),(0,1),...,(0,M-1)
    @testset "Observables: straight rod" begin
        sys = LatticePolymer(; dims=[20, 20], num_poly=1, length_poly=6)
        init!(sys, :ordered)

        # End-to-end: |y=5 - y=0|² = 25
        @test end_to_end_distance_sq(sys, 1) == 25

        # CM: x=0, y=mean(0:5)=2.5
        cm = center_of_mass(sys, 1)
        @test cm[1] ≈ 0.0
        @test cm[2] ≈ 2.5

        # Rg²: variance of {0,1,2,3,4,5} around mean 2.5 = 17.5/6
        @test radius_of_gyration_sq(sys, 1) ≈ 17.5 / 6

        # tr(G) == Rg²
        G = gyration_tensor(sys, 1)
        @test sum(G[d,d] for d in 1:2) ≈ radius_of_gyration_sq(sys, 1)

        # Rod along y → G[1,1]=0, G[2,2]=Rg², off-diag=0
        @test G[1,1] ≈ 0.0 atol=1e-12
        @test G[1,2] ≈ 0.0 atol=1e-12
        @test G[2,2] ≈ radius_of_gyration_sq(sys, 1)
    end

    # ── Cluster observables ──────────────────────────────────────────────────
    @testset "Clusters" begin
        # Well-separated polymers on large lattice → each is its own cluster
        sys = LatticePolymer(; dims=[20, 20], num_poly=2, length_poly=3)
        init!(sys, :ordered)
        c = clusters(sys)
        @test c == [3, 3]  # sorted descending
        @test largest_cluster_size(c) == 3
        @test second_largest_cluster_size(c) == 3
        @test cluster_size_distribution(c) == Dict(3 => 2)

        # Single polymer → one cluster
        sys2 = LatticePolymer(; dims=[10, 10], num_poly=1, length_poly=5)
        init!(sys2, :ordered)
        c2 = clusters(sys2)
        @test c2 == [5]
        @test second_largest_cluster_size(c2) == 0

        # Dense system: clusters merge when polymers touch
        sys3 = LatticePolymer(; dims=[4, 4], num_poly=4, length_poly=3)
        init!(sys3, :random; rng=Xoshiro(42))
        c3 = clusters(sys3)
        @test sum(c3) == 12  # total monomers conserved
        @test largest_cluster_size(c3) == first(c3)
    end
end
