@testset "Periodic Geometry" begin
    L = 10.0

    @testset "wrap_coordinate" begin
        @test wrap_coordinate(5.0, L) ≈ 5.0
        @test wrap_coordinate(12.0, L) ≈ 2.0
        @test wrap_coordinate(-3.0, L) ≈ 7.0
        @test wrap_coordinate(0.0, L) ≈ 0.0
        @test wrap_coordinate(10.0, L) ≈ 0.0
    end

    @testset "wrap_position" begin
        pos = SVector(12.0, -3.0, 25.0)
        w = wrap_position(pos, L)
        @test w[1] ≈ 2.0
        @test w[2] ≈ 7.0
        @test w[3] ≈ 5.0
    end

    @testset "minimum_image_sq" begin
        # Same position
        r1 = SVector(1.0, 1.0, 1.0)
        @test minimum_image_sq(r1, r1, L) ≈ 0.0

        # Nearby
        r2 = SVector(2.0, 1.0, 1.0)
        @test minimum_image_sq(r1, r2, L) ≈ 1.0

        # Across boundary: distance should be 2 not 8
        r3 = SVector(1.0, 1.0, 1.0)
        r4 = SVector(9.0, 1.0, 1.0)
        @test minimum_image_sq(r3, r4, L) ≈ 4.0  # min image dist = 2
    end

    @testset "minimum_image_displacement" begin
        r1 = SVector(1.0, 1.0, 1.0)
        r2 = SVector(9.0, 1.0, 1.0)
        d = minimum_image_displacement(r1, r2, L)
        @test d[1] ≈ 2.0  # 1 - 9 + 10 = 2
        @test d[2] ≈ 0.0
        @test d[3] ≈ 0.0
    end
end
