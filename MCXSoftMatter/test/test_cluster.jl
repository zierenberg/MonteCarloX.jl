@testset "Cluster Analysis" begin

    @testset "All particles far apart" begin
        L = 100.0
        positions = [SVector(10.0*i, 0.0, 0.0) for i in 1:5]
        c = clusters(positions, L, 2.0)
        @test length(c) == 5
        @test all(s -> s == 1, c)
    end

    @testset "All particles close" begin
        L = 100.0
        positions = [SVector(0.0 + 0.1*i, 0.0, 0.0) for i in 1:5]
        c = clusters(positions, L, 2.0)
        @test length(c) == 1
        @test c[1] == 5
    end

    @testset "Two clusters" begin
        L = 100.0
        positions = [
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.0, 0.0),
            SVector(50.0, 0.0, 0.0),
            SVector(51.0, 0.0, 0.0),
        ]
        c = clusters(positions, L, 2.0)
        @test length(c) == 2
        @test sort(c) == [2, 2]
    end

    @testset "Periodic boundary cluster" begin
        L = 10.0
        # Particles at 0.5 and 9.5 should be distance 1 apart via PBC
        positions = [
            SVector(0.5, 0.0, 0.0),
            SVector(9.5, 0.0, 0.0),
        ]
        c = clusters(positions, L, 2.0)
        @test length(c) == 1
    end

    @testset "largest_cluster_size" begin
        @test largest_cluster_size([3, 2]) == 3
        @test largest_cluster_size(Int[]) == 0
    end

    @testset "second_largest_cluster_size" begin
        @test second_largest_cluster_size([3, 2, 1]) == 2
        @test second_largest_cluster_size([2]) == 0
    end

    @testset "cluster_size_distribution" begin
        c = [2, 2, 1]
        dist = cluster_size_distribution(c)
        @test dist[2] == 2
        @test dist[1] == 1
    end
end
