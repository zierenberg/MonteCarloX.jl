"""
    AbstractLatticePolymerSystem

Base type for lattice-based polymer systems on a D-dimensional hypercubic lattice.
"""
abstract type AbstractLatticePolymerSystem <: AbstractSystem end

# ── Coordinate conversion ─────────────────────────────────────────────────────
# Column-major ordering (first dimension has stride 1).

"""
    site_to_coords(site::Int, L::SVector{D,Int}) -> SVector{D,Int}

Convert a 1-based linear site index to 0-based D-dimensional coordinates.
"""
@inline function site_to_coords(site::Int, L::SVector{D,Int}) where {D}
    s = site - 1
    SVector{D,Int}(ntuple(d -> begin
        c = s % L[d]
        s ÷= L[d]
        c
    end, Val(D)))
end

"""
    coords_to_site(coords::SVector{D,Int}, L::SVector{D,Int}) -> Int

Convert 0-based coordinates to a 1-based linear site index.
"""
@inline function coords_to_site(coords::SVector{D,Int}, L::SVector{D,Int}) where {D}
    s = 0
    for d in D:-1:1
        s = s * L[d] + coords[d]
    end
    return s + 1
end

# ── Periodic boundary conditions ──────────────────────────────────────────────

"""
    apply_pbc(x, L) -> Int

Wrap coordinate `x` into [0, L) under periodic boundary conditions.
"""
@inline apply_pbc(x::Int, L::Int) = mod(x, L)

# ── Lattice geometry ──────────────────────────────────────────────────────────

"""
    lattice_difference(c1, c2, L)

Minimum-image displacement c1 - c2 on a periodic axis of length L.
"""
@inline function lattice_difference(c1::Int, c2::Int, L::Int)
    d = mod(c1 - c2 + L, L)
    return d > L >> 1 ? d - L : d
end

@inline function lattice_difference(c1::Float64, c2::Float64, L::Int)
    Lf = Float64(L)
    d = mod(c1 - c2 + Lf, Lf)
    return d > Lf / 2 ? d - Lf : d
end

@inline lattice_difference(c1::Real, c2::Real, L::Int) =
    lattice_difference(Float64(c1), Float64(c2), L)

"""
    lattice_difference(c1, c2, L) -> SVector{D}

Minimum-image displacement vector c1 - c2.
"""
@inline function lattice_difference(c1::SVector{D,Int}, c2::SVector{D,Int}, L::SVector{D,Int}) where {D}
    SVector{D,Int}(ntuple(d -> lattice_difference(c1[d], c2[d], L[d]), Val(D)))
end

@inline function lattice_difference(c1::SVector{D}, c2::SVector{D}, L::SVector{D,Int}) where {D}
    SVector{D,Float64}(ntuple(d -> lattice_difference(c1[d], c2[d], L[d]), Val(D)))
end

# ── Neighbor table ────────────────────────────────────────────────────────────

"""
    _build_lattice_neighbors(dims::SVector{D,Int}) -> Vector{NTuple{2D,Int}}

Build a flat neighbor table for a D-dimensional periodic hypercubic lattice.
Each site has exactly 2D neighbors (compile-time constant), stored inline
in a contiguous `Vector{NTuple{2D,Int}}` — no heap pointer per site.

Ordering: column-major (dimension 1 has stride 1).
Neighbors are sorted by site index to match Graphs.jl ordering for determinism.
"""
function _build_lattice_neighbors(dims::SVector{D,Int}) where D
    N = prod(dims)
    strides = ntuple(d -> d == 1 ? 1 : prod(Tuple(dims)[1:d-1]), Val(D))
    nbrs = Vector{NTuple{2D,Int}}(undef, N)
    for site in 1:N
        s0 = site - 1
        coords = ntuple(d -> (s0 ÷ strides[d]) % dims[d], Val(D))
        # Collect neighbors in (-d1, +d1, -d2, +d2, ...) order, then sort
        neighbors_unsorted = ntuple(Val(2D)) do k
            d = (k + 1) ÷ 2
            dir = iseven(k) ? 1 : -1
            site + (mod(coords[d] + dir, dims[d]) - coords[d]) * strides[d]
        end
        # Sort to match Graphs.jl ordering
        nbrs[site] = ntuple(i -> sort(collect(neighbors_unsorted))[i], Val(2D))
    end
    return nbrs
end

"""
    lattice_distance_sq(c1, c2, L) -> Int

Squared minimum-image distance.
"""
@inline function lattice_distance_sq(c1::SVector{D,Int}, c2::SVector{D,Int}, L::SVector{D,Int}) where {D}
    sum(abs2, lattice_difference(c1, c2, L))
end
