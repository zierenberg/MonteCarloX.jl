using LatticePolymerSystems
using Test

@testset "Cubic Lattice Geometry" begin
    @testset "site_index / site_coords roundtrip" begin
        Lx, Ly, Lz = 4, 5, 6
        for z in 0:Lz-1, y in 0:Ly-1, x in 0:Lx-1
            site = site_index(x, y, z, Lx, Ly)
            @test 1 <= site <= Lx * Ly * Lz
            xr, yr, zr = site_coords(site, Lx, Ly)
            @test (xr, yr, zr) == (x, y, z)
        end
    end

    @testset "site_index uniqueness" begin
        Lx, Ly, Lz = 3, 4, 5
        N = Lx * Ly * Lz
        indices = Set{Int}()
        for z in 0:Lz-1, y in 0:Ly-1, x in 0:Lx-1
            push!(indices, site_index(x, y, z, Lx, Ly))
        end
        @test length(indices) == N
    end

    @testset "build_cubic_neighbors" begin
        L = 4
        nbrs = build_cubic_neighbors(L, L, L)
        @test length(nbrs) == L^3

        # Each site has 6 distinct neighbors
        for site in 1:L^3
            ns = collect(nbrs[site])
            @test length(unique(ns)) == 6
            @test all(1 .<= ns .<= L^3)
            # None of the neighbors should be the site itself
            @test site ∉ ns
        end
    end

    @testset "PBC: corner site neighbors wrap" begin
        L = 4
        nbrs = build_cubic_neighbors(L, L, L)
        # Site (0,0,0) = index 1
        corner = site_index(0, 0, 0, L, L)
        ns = nbrs[corner]
        # +x neighbor should be (1,0,0)
        @test ns[1] == site_index(1, 0, 0, L, L)
        # -x neighbor should wrap to (L-1,0,0)
        @test ns[2] == site_index(L-1, 0, 0, L, L)
        # -y wraps to (0,L-1,0)
        @test ns[4] == site_index(0, L-1, 0, L, L)
        # -z wraps to (0,0,L-1)
        @test ns[6] == site_index(0, 0, L-1, L, L)
    end

    @testset "Neighbor symmetry" begin
        Lx, Ly, Lz = 3, 4, 5
        nbrs = build_cubic_neighbors(Lx, Ly, Lz)
        for site in 1:Lx*Ly*Lz
            for nb in nbrs[site]
                @test site ∈ nbrs[nb]
            end
        end
    end

    @testset "lattice_difference" begin
        L = 10
        @test lattice_difference(1, 9, L) == 2   # 1-9 = -8, wraps to +2
        @test lattice_difference(9, 1, L) == -2   # 9-1 = 8, wraps to -2
        @test lattice_difference(5, 5, L) == 0
        @test lattice_difference(0, 5, L) == -5   # boundary: -5 or +5, round to -5 or 5
    end
end
