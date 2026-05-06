"""
    AbstractLatticePolymerSystem

Base type for lattice-based polymer systems on a D-dimensional hypercubic lattice.
"""
abstract type AbstractLatticePolymerSystem <: AbstractSystem end

# ── Coordinate conversion ─────────────────────────────────────────────────────
# Column-major ordering (first dimension has stride 1), matching Graphs.jl's grid().

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

"""
    lattice_distance_sq(c1, c2, L) -> Int

Squared minimum-image distance.
"""
@inline function lattice_distance_sq(c1::SVector{D,Int}, c2::SVector{D,Int}, L::SVector{D,Int}) where {D}
    sum(abs2, lattice_difference(c1, c2, L))
end
