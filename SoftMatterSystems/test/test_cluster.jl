@testset "Cluster Analysis" begin

    @testset "All particles far apart" begin
        L = 100.0
        positions = [SVector(10.0*i, 0.0, 0.0) for i in 1:5]
        clusters = geometric_clusters(positions, L, 2.0)
        @test length(clusters) == 5
        @test all(c -> length(c) == 1, clusters)
    end

    @testset "All particles close" begin
        L = 100.0
        positions = [SVector(0.0 + 0.1*i, 0.0, 0.0) for i in 1:5]
        clusters = geometric_clusters(positions, L, 2.0)
        @test length(clusters) == 1
        @test length(clusters[1]) == 5
    end

    @testset "Two clusters" begin
        L = 100.0
        positions = [
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.0, 0.0),
            SVector(50.0, 0.0, 0.0),
            SVector(51.0, 0.0, 0.0),
        ]
        clusters = geometric_clusters(positions, L, 2.0)
        @test length(clusters) == 2
        sizes = sort([length(c) for c in clusters])
        @test sizes == [2, 2]
    end

    @testset "Periodic boundary cluster" begin
        L = 10.0
        # Particles at 0.5 and 9.5 should be distance 1 apart via PBC
        positions = [
            SVector(0.5, 0.0, 0.0),
            SVector(9.5, 0.0, 0.0),
        ]
        clusters = geometric_clusters(positions, L, 2.0)
        @test length(clusters) == 1
    end

    @testset "largest_cluster_size" begin
        clusters = [[1, 2, 3], [4, 5]]
        @test largest_cluster_size(clusters) == 3
        @test largest_cluster_size(Vector{Vector{Int}}()) == 0
    end

    @testset "second_largest_cluster_size" begin
        clusters = [[1, 2, 3], [4, 5], [6]]
        @test second_largest_cluster_size(clusters) == 2
        @test second_largest_cluster_size([[1, 2]]) == 0
    end

    @testset "cluster_size_distribution" begin
        clusters = [[1, 2], [3, 4], [5]]
        dist = cluster_size_distribution(clusters)
        @test dist[2] == 2
        @test dist[1] == 1
    end
end
