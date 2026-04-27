using LatticePolymerSystems
using Test

@testset "Cluster Analysis" begin
    @testset "No occupied sites" begin
        nbrs = build_cubic_neighbors(3, 3, 3)
        occupation = zeros(Bool, 27)
        clusters = flood_fill_clusters(occupation, nbrs)
        @test isempty(clusters)
        @test largest_cluster_size(clusters) == 0
    end

    @testset "Single particle" begin
        nbrs = build_cubic_neighbors(3, 3, 3)
        occupation = zeros(Bool, 27)
        occupation[1] = true
        clusters = flood_fill_clusters(occupation, nbrs)
        @test length(clusters) == 1
        @test largest_cluster_size(clusters) == 1
    end

    @testset "Two separate particles" begin
        L = 6
        nbrs = build_cubic_neighbors(L, L, L)
        occupation = zeros(Bool, L^3)
        # Place two particles far apart (not neighbors)
        s1 = site_index(0, 0, 0, L, L)
        s2 = site_index(3, 3, 3, L, L)
        occupation[s1] = true
        occupation[s2] = true
        clusters = flood_fill_clusters(occupation, nbrs)
        @test length(clusters) == 2
        @test largest_cluster_size(clusters) == 1
        @test second_largest_cluster_size(clusters) == 1
    end

    @testset "Two adjacent particles form one cluster" begin
        L = 4
        nbrs = build_cubic_neighbors(L, L, L)
        occupation = zeros(Bool, L^3)
        s1 = site_index(0, 0, 0, L, L)
        s2 = site_index(1, 0, 0, L, L)
        occupation[s1] = true
        occupation[s2] = true
        clusters = flood_fill_clusters(occupation, nbrs)
        @test length(clusters) == 1
        @test largest_cluster_size(clusters) == 2
    end

    @testset "PBC: wrap-around cluster" begin
        L = 4
        nbrs = build_cubic_neighbors(L, L, L)
        occupation = zeros(Bool, L^3)
        # (0,0,0) and (L-1,0,0) are neighbors via PBC
        s1 = site_index(0, 0, 0, L, L)
        s2 = site_index(L-1, 0, 0, L, L)
        occupation[s1] = true
        occupation[s2] = true
        clusters = flood_fill_clusters(occupation, nbrs)
        @test length(clusters) == 1
        @test largest_cluster_size(clusters) == 2
    end

    @testset "Cluster size distribution" begin
        L = 4
        nbrs = build_cubic_neighbors(L, L, L)
        occupation = zeros(Bool, L^3)
        # One cluster of size 3 (line along x)
        occupation[site_index(0, 0, 0, L, L)] = true
        occupation[site_index(1, 0, 0, L, L)] = true
        occupation[site_index(2, 0, 0, L, L)] = true
        # One isolated particle
        occupation[site_index(0, 2, 2, L, L)] = true

        clusters = flood_fill_clusters(occupation, nbrs)
        dist = cluster_size_distribution(clusters)
        @test dist[3] == 1
        @test dist[1] == 1
        @test largest_cluster_size(clusters) == 3
        @test second_largest_cluster_size(clusters) == 1
    end
end
