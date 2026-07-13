# Lattice index structures cross-validated against Graphs.jl (construction-time helpers:
# correctness over speed).

import Graphs

@testset "Lattice neighbor tables vs Graphs.grid(periodic=true)" begin
    for dims in ((5,), (4, 5), (3, 4, 5), (3, 3, 3, 3))
        nbrs = MCXSpins._build_lattice_neighbors(dims)
        g = Graphs.SimpleGraphs.grid(collect(dims); periodic=true)
        @test length(nbrs) == Graphs.nv(g)
        @test all(i -> sort(collect(nbrs[i])) == sort(Graphs.neighbors(g, i)),
                  1:length(nbrs))
    end

    # L = 2 periodic direction: + and − neighbor coincide — the PBC convention counts the
    # bond twice; a SimpleGraph cannot (this is why the table is not built from Graphs.jl).
    nbrs2 = MCXSpins._build_lattice_neighbors((2, 3))
    @test count(==(2), nbrs2[1]) == 2
end

@testset "Oriented partner tables: round-trips and consistency" begin
    for dims in ((4, 5), (3, 4, 5))
        pos, neg = MCXSpins.oriented_partners(dims)
        nbrs = MCXSpins._build_lattice_neighbors(dims)
        D, N = length(dims), prod(dims)
        # step −d then +d returns home (and vice versa)
        @test all(pos[neg[i][d]][d] == i && neg[pos[i][d]][d] == i for i in 1:N, d in 1:D)
        @test all(sort([pos[i]..., neg[i]...]) == sort(collect(nbrs[i])) for i in 1:N)
    end
end
