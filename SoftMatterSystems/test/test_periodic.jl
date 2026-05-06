using StaticArrays: SVector

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
        @test wrap_position(SVector(12.0, -3.0, 25.0), L) ≈ SVector(2.0, 7.0, 5.0)
        @test wrap_position(SVector(12.0, -3.0), L) ≈ SVector(2.0, 7.0)  # 2D
    end

    @testset "minimum_image_sq" begin
        r1 = SVector(1.0, 1.0, 1.0)
        @test minimum_image_sq(r1, r1, L) ≈ 0.0
        @test minimum_image_sq(r1, SVector(2.0, 1.0, 1.0), L) ≈ 1.0
        # Across boundary: min-image distance is 2, not 8
        @test minimum_image_sq(r1, SVector(9.0, 1.0, 1.0), L) ≈ 4.0
        # 2D
        @test minimum_image_sq(SVector(1.0, 1.0), SVector(9.0, 1.0), L) ≈ 4.0
    end

    @testset "minimum_image_displacement" begin
        d = minimum_image_displacement(SVector(1.0, 1.0, 1.0), SVector(9.0, 1.0, 1.0), L)
        @test d ≈ SVector(2.0, 0.0, 0.0)
    end
end
