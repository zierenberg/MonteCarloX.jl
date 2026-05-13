using StaticArrays: SVector

@testset "Periodic Geometry" begin
    box = PeriodicBox{3}(10.0)

    @testset "PeriodicBox construction" begin
        @test box.L ≈ SVector(10.0, 10.0, 10.0)
        @test box.inv_L ≈ SVector(0.1, 0.1, 0.1)

        # Anisotropic box
        abox = PeriodicBox(SVector(10.0, 20.0, 5.0))
        @test abox.L ≈ SVector(10.0, 20.0, 5.0)
        @test abox.inv_L ≈ SVector(0.1, 0.05, 0.2)
    end

    @testset "constrain" begin
        @test constrain(box, SVector(12.0, -3.0, 25.0)) ≈ SVector(2.0, 7.0, 5.0)
        # 2D
        box2d = PeriodicBox{2}(10.0)
        @test constrain(box2d, SVector(12.0, -3.0)) ≈ SVector(2.0, 7.0)
    end

    @testset "distance_sq" begin
        r1 = SVector(1.0, 1.0, 1.0)
        @test distance_sq(box, r1, r1) ≈ 0.0
        @test distance_sq(box, r1, SVector(2.0, 1.0, 1.0)) ≈ 1.0
        # Across boundary: min-image distance is 2, not 8
        @test distance_sq(box, r1, SVector(9.0, 1.0, 1.0)) ≈ 4.0
        # 2D
        box2d = PeriodicBox{2}(10.0)
        @test distance_sq(box2d, SVector(1.0, 1.0), SVector(9.0, 1.0)) ≈ 4.0
    end

    @testset "difference" begin
        d = difference(box, SVector(1.0, 1.0, 1.0), SVector(9.0, 1.0, 1.0))
        @test d ≈ SVector(2.0, 0.0, 0.0)
    end

    @testset "anisotropic distance_sq" begin
        abox = PeriodicBox(SVector(10.0, 20.0, 5.0))
        # x-direction: wrap across L=10
        @test distance_sq(abox, SVector(1.0, 0.0, 0.0), SVector(9.0, 0.0, 0.0)) ≈ 4.0
        # y-direction: no wrap at distance 8 (L=20, so direct distance is shorter)
        @test distance_sq(abox, SVector(0.0, 1.0, 0.0), SVector(0.0, 9.0, 0.0)) ≈ 64.0
        # z-direction: wrap across L=5
        @test distance_sq(abox, SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 4.0)) ≈ 4.0
    end
end
