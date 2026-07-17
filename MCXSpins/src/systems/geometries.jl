# ── Geometries: index structures fed to the interactions ──────────────────────
#
# A system's topology argument is a dims vector (periodic hypercubic lattice), a Graphs.jl
# `SimpleGraph`, or a sparse J_ij matrix — resolved by dispatch in the model constructors.
# We do NOT reinvent graph machinery: arbitrary connectivity is Graphs.jl's job
# (`graph_partners` is a one-line adapter). What lives here are only the two
# performance/physics specializations Graphs.jl cannot provide:
#   • fixed-degree NTuple neighbor tables for periodic hypercubic lattices (compile-time
#     degree → the PairInteraction fast path), including the direction-resolved variant the
#     vision-cone interaction needs;
#   • sparse J_ij builders where the VALUES (couplings), not the connectivity, are the point.

"""
    _build_lattice_neighbors(dims::NTuple{D,Int}) -> Vector{NTuple{2D,Int}}

Neighbor table for a D-dimensional periodic hypercubic lattice. Each site has exactly 2D
neighbors (compile-time known → fixed-degree fast path in `PairInteraction`).

Hand-rolled rather than converted from `Graphs.grid(dims; periodic=true)` for one
correctness reason, not speed: a `SimpleGraph` cannot represent the DOUBLE bond of a
periodic direction with L = 2 (the + and − neighbor coincide; the standard PBC convention
counts that bond twice, a simple graph collapses it to one edge). The table below lists it
twice. Cross-validated against `Graphs.grid` for L ≥ 3 in the test suite.
"""
function _build_lattice_neighbors(dims::NTuple{D,Int}) where D
    N = prod(dims)
    strides = ntuple(d -> d == 1 ? 1 : prod(dims[1:d-1]), Val(D))
    nbrs = Vector{NTuple{2D,Int}}(undef, N)
    for site in 1:N
        s0 = site - 1
        coords = ntuple(d -> (s0 ÷ strides[d]) % dims[d], Val(D))
        nbrs[site] = ntuple(Val(2D)) do k
            d = (k + 1) ÷ 2
            dir = iseven(k) ? 1 : -1
            site + (mod(coords[d] + dir, dims[d]) - coords[d]) * strides[d]
        end
    end
    return nbrs
end

# Oriented neighbor tables: `pos[i][d]` is site i's neighbor in the +direction of axis d
# (right, up, …), `neg[i][d]` in the −direction (left, down, …). Direction-resolved
# adjacency for the vision-cone interaction. Graphs.jl cannot provide this: a graph stores
# edges without axis/direction labels, so recovering "which neighbor is +x" needs the same
# index arithmetic anyway. Robustness comes from tests: round-trips (pos∘neg = id) and
# consistency with _build_lattice_neighbors.
function oriented_partners(dims::NTuple{D,Int}) where D
    N = prod(dims)
    strides = ntuple(d -> d == 1 ? 1 : prod(dims[1:d-1]), Val(D))
    pos = Vector{NTuple{D,Int}}(undef, N)
    neg = Vector{NTuple{D,Int}}(undef, N)
    for site in 1:N
        s0 = site - 1
        coords = ntuple(d -> (s0 ÷ strides[d]) % dims[d], Val(D))
        pos[site] = ntuple(d -> site + (mod(coords[d] + 1, dims[d]) - coords[d]) * strides[d], Val(D))
        neg[site] = ntuple(d -> site + (mod(coords[d] - 1, dims[d]) - coords[d]) * strides[d], Val(D))
    end
    return pos, neg
end

# Variable-degree adjacency lists from an arbitrary graph — the Vector{Vector{Int}} storage
# layout of `PairInteraction`. Everything graph-topological beyond this line belongs to
# Graphs.jl, not to MCXSpins.
graph_partners(g::SimpleGraph) = [collect(Graphs.neighbors(g, i)) for i in 1:nv(g)]

# Topology arguments of the model constructors — construction-time only, never in the
# dynamics (the site count follows as length(partner table), so no separate helper).
_init_partners(dims::AbstractVector{<:Integer}) = _build_lattice_neighbors(Tuple(Int.(dims)))
_init_partners(g::SimpleGraph) = graph_partners(g)

_geometry(dims::AbstractVector{<:Integer}) = Tuple(Int.(dims))
_geometry(g::SimpleGraph) = g

"""
    lattice_random_J(dims, rng; dist=randn, directed=false)

Random-J sparse coupling matrix on the bonds of a periodic hypercubic lattice (EA spin
glasses, cross-checks). `directed=true` draws J_ij and J_ji independently → asymmetric
couplings (no Hamiltonian).
"""
function lattice_random_J(dims::NTuple{D,Int}, rng::AbstractRNG; dist=randn, directed=false) where D
    partners = _build_lattice_neighbors(dims)
    N = prod(dims)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:N, j in partners[i]
        j > i || continue
        Jij = Float64(dist(rng))
        Jji = directed ? Float64(dist(rng)) : Jij
        push!(rows, i); push!(cols, j); push!(vals, Jij)
        push!(rows, j); push!(cols, i); push!(vals, Jji)
    end
    return sparse(rows, cols, vals, N, N)
end

# Hopfield couplings J_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ (zero diagonal) from patterns ξ (N×P).
# Patterns are quenched parameters fixed at construction, like J or Δ.
function hopfield_J(patterns::AbstractMatrix{<:Integer})
    N, P = size(patterns)
    Jd = zeros(N, N)
    for μ in 1:P, j in 1:N, i in 1:N
        i != j && (Jd[i, j] += patterns[i, μ] * patterns[j, μ] / N)
    end
    return sparse(Jd)
end
