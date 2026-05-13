"""
    LatticePolymer{D, TJ<:Real} <: AbstractLatticePolymerSystem

D-dimensional lattice polymer system on a hypercubic lattice with PBC.

# Type parameters
- `D`: spatial dimension (compile-time for coordinate loop unrolling)
- `TJ`: coupling type (Float64)

# Energy (all nearest-neighbor contacts, including backbone)

    E = -J_intra * (intra-polymer contacts) - J_inter * (inter-polymer contacts)

Backbone bonds contribute a fixed -J_intra each and are included.
"""
mutable struct LatticePolymer{D, K, TJ<:Real} <: AbstractLatticePolymerSystem
    polymers::Vector{Vector{SVector{D,Int}}}
    neighbors::Vector{NTuple{K,Int}} # flat table: K=2D neighbors per site, no heap ptr
    dims::SVector{D,Int}
    state::Vector{Int}               # state[site] = polymer ID (0 = empty)
    J_intra::TJ
    J_inter::TJ
    cached_intra::Int                # total intra-polymer contact count
    cached_inter::Int                # total inter-polymer contact count
end

# ── Constructors ─────────────────────────────────────────────────────────────

"""
    LatticePolymer(; dims, polys, J_intra=0.0, J_inter=1.0)

Primary constructor. `dims` is the box size per dimension, `polys` is a vector of
polymer lengths.

    LatticePolymer(; dims, num_poly, length_poly, J_intra=0.0, J_inter=1.0)

Convenience for `num_poly` homopolymers of uniform `length_poly`.
"""
function LatticePolymer(; dims::AbstractVector{<:Integer},
                          polys::Union{AbstractVector{<:Integer}, Nothing}=nothing,
                          num_poly::Union{Integer, Nothing}=nothing,
                          length_poly::Union{Integer, Nothing}=nothing,
                          J_intra::Real=0.0, J_inter::Real=1.0)
    if polys !== nothing
        lengths = collect(Int, polys)
    else
        @assert num_poly !== nothing && length_poly !== nothing "Provide either `polys` or both `num_poly` and `length_poly`"
        lengths = fill(Int(length_poly), Int(num_poly))
    end
    D = length(dims)
    sv_dims = SVector{D,Int}(dims...)
    N_sites = prod(sv_dims)
    @assert sum(lengths) <= N_sites "Not enough sites for $(length(lengths)) polymers ($(sum(lengths)) monomers on $N_sites sites)"
    nbrs = _build_lattice_neighbors(sv_dims)
    LatticePolymer{D, 2*D, Float64}(
        [Vector{SVector{D,Int}}([zero(SVector{D,Int}) for _ in 1:l]) for l in lengths],
        nbrs, sv_dims,
        zeros(Int, N_sites),
        Float64(J_intra), Float64(J_inter),
        0, 0
    )
end

# ── Helpers ───────────────────────────────────────────────────────────────────

num_polymers(sys::LatticePolymer) = length(sys.polymers)
polymer_length(sys::LatticePolymer, n::Int) = length(sys.polymers[n])

# ── Contact counting ─────────────────────────────────────────────────────────

"""
    site_contacts(sys, site) -> (intra::Int, inter::Int)

Count intra- and inter-polymer contacts at `site`. Pure integer arithmetic.
"""
@inline function site_contacts(sys::LatticePolymer, site::Int)
    owner = sys.state[site]
    owner == 0 && return (0, 0)
    intra, inter = 0, 0
    @inbounds for nb in sys.neighbors[site]
        occ = sys.state[nb]
        occ == 0 && continue
        if occ == owner
            intra += 1
        else
            inter += 1
        end
    end
    return (intra, inter)
end

"""
    site_energy(sys, site)

Energy contribution at `site`: `-J_intra * n_intra - J_inter * n_inter`.
"""
@inline function site_energy(sys::LatticePolymer, site::Int)
    ni, ne = site_contacts(sys, site)
    return -sys.J_intra * ni - sys.J_inter * ne
end

# ── Energy ───────────────────────────────────────────────────────────────────

function _recompute_contacts!(sys::LatticePolymer)
    intra, inter = 0, 0
    @inbounds for site in eachindex(sys.state)
        ci, ce = site_contacts(sys, site)
        intra += ci
        inter += ce
    end
    sys.cached_intra = intra ÷ 2
    sys.cached_inter = inter ÷ 2
    return nothing
end

function energy(sys::LatticePolymer; full::Bool=false)
    full && _recompute_contacts!(sys)
    return -sys.J_intra * sys.cached_intra - sys.J_inter * sys.cached_inter
end

# ── Initialization ────────────────────────────────────────────────────────────

function init!(sys::LatticePolymer{D}, type::Symbol; rng=nothing) where {D}
    fill!(sys.state, 0)

    if type == :ordered
        _init_ordered!(sys)
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        _init_random!(sys, rng)
    else
        error("Unknown initialization type: $type")
    end

    _recompute_contacts!(sys)
    return sys
end

function _init_ordered!(sys::LatticePolymer{D}) where {D}
    @assert D >= 2 "Ordered init requires D >= 2"
    N = num_polymers(sys)
    spacing = sys.dims[1] ÷ N
    @assert spacing >= 1 "Not enough x-space for $N polymers"
    for n in 1:N
        M = polymer_length(sys, n)
        @assert M <= sys.dims[2] "Polymer $n length $M exceeds y-dimension $(sys.dims[2])"
        x = (n - 1) * spacing
        for m in 1:M
            c = SVector{D,Int}(ntuple(d -> d == 1 ? x : d == 2 ? m - 1 : 0, Val(D)))
            sys.polymers[n][m] = c
            sys.state[coords_to_site(c, sys.dims)] = n
        end
    end
end

function _init_random!(sys::LatticePolymer{D}, rng) where {D}
    for n in 1:num_polymers(sys)
        M = polymer_length(sys, n)
        success = false
        for _ in 1:1000
            _reset_polymer!(sys, n)
            if _grow_saw!(sys, n, M, rng)
                success = true
                break
            end
        end
        success || error("Failed to place polymer $n after 1000 attempts")
    end
end

function _reset_polymer!(sys::LatticePolymer, n::Int)
    for c in sys.polymers[n]
        s = coords_to_site(c, sys.dims)
        if sys.state[s] == n
            sys.state[s] = 0
        end
    end
end

function _grow_saw!(sys::LatticePolymer{D}, n::Int, M::Int, rng) where {D}
    empty_sites = findall(iszero, sys.state)
    isempty(empty_sites) && return false
    start_site = rand(rng, empty_sites)
    sys.state[start_site] = n
    sys.polymers[n][1] = site_to_coords(start_site, sys.dims)

    for m in 2:M
        last_site = coords_to_site(sys.polymers[n][m-1], sys.dims)
        nbrs = MVector(sys.neighbors[last_site])  # stack-allocated mutable copy for shuffle
        for i in length(nbrs):-1:2
            j = rand(rng, 1:i)
            nbrs[i], nbrs[j] = nbrs[j], nbrs[i]
        end
        placed = false
        for nb_site in nbrs
            if sys.state[nb_site] == 0
                sys.state[nb_site] = n
                sys.polymers[n][m] = site_to_coords(nb_site, sys.dims)
                placed = true
                break
            end
        end
        if !placed
            _reset_polymer!(sys, n)
            return false
        end
    end
    return true
end
