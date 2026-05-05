"""
    AbstractLatticePolymerSystem

Base type for lattice-based polymer systems on a D-dimensional hypercubic lattice.
"""
abstract type AbstractLatticePolymerSystem <: AbstractSystem end

# ── Coordinate conversion ─────────────────────────────────────────────────────
# Column-major ordering (first dimension has stride 1), matching Graphs.jl's grid().

"""
    site_to_coords(site::Int, L::SVector{D,Int}) -> MVector{D,Int}

Convert a 1-based linear site index to 0-based D-dimensional coordinates.
"""
@inline function site_to_coords(site::Int, L::SVector{D,Int}) where {D}
    s = site - 1
    result = MVector{D,Int}(undef)
    for d in 1:D
        result[d] = s % L[d]
        s ÷= L[d]
    end
    return result
end

"""
    site_to_coords!(result::MVector{D,Int}, site, L)

In-place version: writes coordinates into `result`, zero allocations.
"""
@inline function site_to_coords!(result::MVector{D,Int}, site::Int, L::SVector{D,Int}) where {D}
    s = site - 1
    for d in 1:D
        result[d] = s % L[d]
        s ÷= L[d]
    end
    return result
end

"""
    coords_to_site(coords::MVector{D,Int}, L::SVector{D,Int}) -> Int

Convert 0-based coordinates to a 1-based linear site index.
"""
@inline function coords_to_site(coords::MVector{D,Int}, L::SVector{D,Int}) where {D}
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
    lattice_difference(c1, c2, L) -> MVector{D}

Minimum-image displacement vector c1 - c2.
"""
@inline function lattice_difference(c1::MVector{D,Int}, c2::MVector{D,Int}, L::SVector{D,Int}) where {D}
    d = MVector{D,Int}(undef)
    for dd in 1:D
        d[dd] = lattice_difference(c1[dd], c2[dd], L[dd])
    end
    return d
end

@inline function lattice_difference(c1::MVector{D}, c2::MVector{D}, L::SVector{D,Int}) where {D}
    d = MVector{D,Float64}(undef)
    for dd in 1:D
        d[dd] = lattice_difference(c1[dd], c2[dd], L[dd])
    end
    return d
end

"""
    lattice_distance_sq(c1, c2, L) -> Int

Squared minimum-image distance.
"""
@inline function lattice_distance_sq(c1::MVector{D,Int}, c2::MVector{D,Int}, L::SVector{D,Int}) where {D}
    d = lattice_difference(c1, c2, L)
    return sum(abs2, d)
end
