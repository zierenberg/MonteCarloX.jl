"""
    AbstractSpinSystem <: AbstractSystem

Base type for spin systems.
"""
abstract type AbstractSpinSystem <: AbstractSystem end

"""
    delta_sys(sys, i, s_new)

Prepare a system-specific local move delta payload for site `i` and proposal `s_new`.
Default behavior is identity (`s_new`), so existing models remain compatible.
"""
@inline delta_sys(sys::AbstractSpinSystem, i, s_new) = s_new

"""
    NoField

Sentinel type indicating no external field. Zero-cost at runtime.
"""
struct NoField end

"""
    pick_site(rng, N)

Randomly pick a site index from 1 to N.
"""
@inline pick_site(rng, N) = Int(rand(rng, UInt) % UInt(N)) + 1

# -- Field helpers ------------------------------------------------------------

@inline _site_field(::NoField, i) = 0
@inline _site_field(h::Real, i) = h
@inline _site_field(h::AbstractVector, i) = @inbounds h[i]

@inline _field_sum(::NoField, spins) = 0.0
@inline _field_sum(h::Real, spins) = float(h) * sum(spins)
@inline function _field_sum(h::AbstractVector, spins)
    s = 0.0
    @inbounds for i in eachindex(spins)
        s += h[i] * spins[i]
    end
    return s
end

# -- Cached field energy: derive from cached_mag when possible ----------------

@inline _cached_field_energy(::NoField, cached_mag, cached_field) = 0.0
@inline _cached_field_energy(h::Real, cached_mag, cached_field) = float(h) * cached_mag
@inline _cached_field_energy(::AbstractVector, cached_mag, cached_field) = cached_field

# -- Field cache update in modify!: only needed for vector fields -------------

@inline _update_field_cache!(sys, ::NoField, i, delta) = nothing
@inline _update_field_cache!(sys, ::Real, i, delta) = nothing
@inline _update_field_cache!(sys, h::AbstractVector, i, delta) = (sys.cached_field += Float64(@inbounds h[i]) * delta)

# -- Shared lattice neighbor builder ------------------------------------------

"""
    _build_lattice_neighbors(dims::NTuple{D,Int}) -> Vector{NTuple{2D,Int}}

Build neighbor table for a D-dimensional periodic hypercubic lattice.
Each site has exactly 2D neighbors (compile-time known).
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

# -- Shared sparse matrix check -----------------------------------------------

function _check_symmetric(J::SparseMatrixCSC)
    n = size(J, 1)
    @assert size(J, 2) == n "J must be square"
    for col in 1:n
        for ptr in J.colptr[col]:(J.colptr[col+1]-1)
            row = J.rowval[ptr]
            row != col && @assert J[row, col] == J[col, row] "J must be symmetric"
        end
    end
end
