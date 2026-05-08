"""
    BeadSpringPolymer{D, T, TPair, TBond, TBend, TCell, TIdx} <: AbstractSoftMatterSystem

D-dimensional bead-spring polymer system in a cubic box with PBC.
Heterogeneous chain lengths are supported.

The interaction is decomposed into composable potentials:
- `pair_potential` -- non-bonded pair interaction (e.g. LJ)
- `bond_potential` -- covalent bond along backbone (e.g. FENE)
- `bending_potential` -- bending stiffness at chain angles (e.g. cosine)

Positions are stored flat. For polymer m, monomer k lives at global index
`offsets[m] + k`. `polymer_id[i]` and `monomer_k[i]` give O(1) reverse lookup.
The element type `TIdx` is chosen at construction time (`Int8`/`Int16`/`Int32`)
based on `max(num_poly, maximum(lengths))`.

When the pair potential has a finite cutoff, a cell list is automatically
created for O(N) pair energy evaluation.

# Constructor (uniform lengths)
    BeadSpringPolymer(; D=3, num_poly, length_poly, L, pair_potential, bond_potential,
                        bending_potential=NoBendingPotential())

# Constructor (heterogeneous lengths)
    BeadSpringPolymer(; D=3, num_poly, lengths, L, pair_potential, bond_potential,
                        bending_potential=NoBendingPotential())
"""
mutable struct BeadSpringPolymer{D, T<:AbstractFloat,
                                  TPair<:AbstractPairPotential,
                                  TBond<:AbstractBondPotential,
                                  TBend<:AbstractBendingPotential,
                                  TCell,
                                  TIdx<:Integer} <: AbstractSoftMatterSystem
    positions::Vector{SVector{D,T}}
    num_poly::Int
    lengths::Vector{Int}      # lengths[m]  = number of monomers in polymer m
    offsets::Vector{Int}      # offsets[m]  = global index of first monomer of m, minus 1
    L::T
    pair_potential::TPair
    bond_potential::TBond
    bending_potential::TBend
    cached_energy::T
    cell_list::TCell
    polymer_id::Vector{TIdx}  # polymer_id[i] = polymer index of monomer i
    monomer_k::Vector{TIdx}   # monomer_k[i]  = position of monomer i within its polymer
end

# ── Index type selection ──────────────────────────────────────────────────────

@inline function _index_type(maxval::Int)
    maxval <= typemax(Int8)  && return Int8
    maxval <= typemax(Int16) && return Int16
    return Int32
end

# ── Constructor ──────────────────────────────────────────────────────────────

function BeadSpringPolymer(; D::Int=3,
                             num_poly::Integer,
                             length_poly::Union{Integer,Nothing}=nothing,
                             lengths::Union{AbstractVector{<:Integer},Nothing}=nothing,
                             L,
                             pair_potential::AbstractPairPotential,
                             bond_potential::AbstractBondPotential,
                             bending_potential::AbstractBendingPotential=NoBendingPotential())
    @assert (length_poly !== nothing) ⊻ (lengths !== nothing) "Provide either `length_poly` or `lengths`"
    num_poly_i = Int(num_poly)
    lens = length_poly !== nothing ? fill(Int(length_poly), num_poly_i) : Int.(lengths)
    @assert length(lens) == num_poly_i "lengths must have length num_poly"

    T = typeof(float(L))
    offs = Vector{Int}(undef, num_poly_i)
    offs[1] = 0
    for m in 2:num_poly_i
        offs[m] = offs[m-1] + lens[m-1]
    end
    n_total = offs[end] + lens[end]

    TIdx = _index_type(max(num_poly_i, maximum(lens)))
    poly_id = Vector{TIdx}(undef, n_total)
    mono_k  = Vector{TIdx}(undef, n_total)
    for m in 1:num_poly_i
        for k in 1:lens[m]
            i = offs[m] + k
            poly_id[i] = TIdx(m)
            mono_k[i]  = TIdx(k)
        end
    end

    positions = [zero(SVector{D,T}) for _ in 1:n_total]
    cl = _make_cell_list(Val(D), n_total, T(L), pair_potential)
    BeadSpringPolymer{D, T, typeof(pair_potential), typeof(bond_potential),
                      typeof(bending_potential), typeof(cl), TIdx}(
        positions, num_poly_i, lens, offs, T(L),
        pair_potential, bond_potential, bending_potential,
        zero(T), cl, poly_id, mono_k)
end

# ── Accessors ────────────────────────────────────────────────────────────────

num_polymers(sys::BeadSpringPolymer)          = sys.num_poly
polymer_length(sys::BeadSpringPolymer)        = sys.lengths[1]
polymer_length(sys::BeadSpringPolymer, m::Int) = sys.lengths[m]
total_monomers(sys::BeadSpringPolymer)        = length(sys.positions)

# Global index of the k-th monomer (1-based) of polymer m (1-based)
@inline _monomer_idx(sys::BeadSpringPolymer, m::Int, k::Int) = sys.offsets[m] + k

# ── Initialization ───────────────────────────────────────────────────────────

function init!(sys::BeadSpringPolymer{D,T}, type::Symbol; rng=nothing) where {D,T}
    if type == :random_walk
        @assert rng !== nothing "Random walk initialization requires rng"
        _init_random_walk!(sys, rng)
    else
        error("Unknown initialization type: $type")
    end
    build!(sys.cell_list, sys.positions)
    _recompute_energy!(sys)
    return sys
end

function _init_random_walk!(sys::BeadSpringPolymer{D,T}, rng) where {D,T}
    for m in 1:sys.num_poly
        off = sys.offsets[m]
        M   = sys.lengths[m]
        pos = SVector{D,T}(ntuple(_ -> rand(rng, T) * sys.L, Val(D)))
        sys.positions[off + 1] = pos
        for k in 2:M
            step = _random_unit_vector(Val(D), T, rng)
            pos  = wrap_position(pos + step, sys.L)
            sys.positions[off + k] = pos
        end
    end
end

@inline function _random_unit_vector(::Val{2}, ::Type{T}, rng) where T
    phi = T(2pi) * rand(rng, T)
    SVector{2,T}(cos(phi), sin(phi))
end

@inline function _random_unit_vector(::Val{3}, ::Type{T}, rng) where T
    theta = acos(T(2) * rand(rng, T) - one(T))
    phi = T(2pi) * rand(rng, T)
    SVector{3,T}(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
end

# ── Energy ───────────────────────────────────────────────────────────────────

function _recompute_energy!(sys::BeadSpringPolymer{D,T}) where {D,T}
    sys.cached_energy = _compute_pair_energy(sys) +
                        _compute_bond_energy(sys) +
                        _compute_bending_energy(sys)
    return nothing
end

function energy(sys::BeadSpringPolymer; full::Bool=false)
    full && _recompute_energy!(sys)
    return sys.cached_energy
end

energy_pair(sys::BeadSpringPolymer) = _compute_pair_energy(sys)
energy_bond(sys::BeadSpringPolymer) = _compute_bond_energy(sys)
energy_bending(sys::BeadSpringPolymer) = _compute_bending_energy(sys)

function _compute_pair_energy(sys::BeadSpringPolymer{D,T}) where {D,T}
    E = zero(T)
    @inbounds for m in 1:sys.num_poly
        off = sys.offsets[m]
        M   = sys.lengths[m]
        # Intra-polymer: skip covalent neighbors (ki, ki+1)
        for ki in 1:M-2
            i = off + ki
            for kj in ki+2:M
                j = off + kj
                r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                E += sys.pair_potential(r_sq)
            end
        end
        # Inter-polymer
        for m2 in m+1:sys.num_poly
            off2 = sys.offsets[m2]
            M2   = sys.lengths[m2]
            for ki in 1:M
                i = off + ki
                for kj in 1:M2
                    j = off2 + kj
                    r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                    E += sys.pair_potential(r_sq)
                end
            end
        end
    end
    return E
end

function _compute_bond_energy(sys::BeadSpringPolymer{D,T}) where {D,T}
    sys.bond_potential isa NoBondPotential && return zero(T)
    E = zero(T)
    @inbounds for m in 1:sys.num_poly
        off = sys.offsets[m]
        M   = sys.lengths[m]
        for k in 1:M-1
            r_sq = minimum_image_sq(sys.positions[off+k], sys.positions[off+k+1], sys.L)
            E += sys.bond_potential(r_sq)
        end
    end
    return E
end

"""
    _cos_angle(a, b, c, L) -> T

Cosine of the angle at b formed by a-b-c, using minimum image convention.
"""
@inline function _cos_angle(a::SVector{D,T}, b::SVector{D,T}, c::SVector{D,T}, L) where {D,T}
    ba = minimum_image_displacement(a, b, L)
    bc = minimum_image_displacement(c, b, L)
    return sum(ba .* bc) / (sqrt(sum(abs2, ba)) * sqrt(sum(abs2, bc)))
end

function _compute_bending_energy(sys::BeadSpringPolymer{D,T}) where {D,T}
    sys.bending_potential isa NoBendingPotential && return zero(T)
    E = zero(T)
    @inbounds for m in 1:sys.num_poly
        off = sys.offsets[m]
        M   = sys.lengths[m]
        for k in 1:M-2
            cos_theta = _cos_angle(sys.positions[off+k], sys.positions[off+k+1], sys.positions[off+k+2], sys.L)
            E += sys.bending_potential(cos_theta)
        end
    end
    return E
end

# ── Local energy for a single monomer ────────────────────────────────────────

"""
    _monomer_energy(sys, idx) -> T

Total energy contribution of monomer at global index `idx`.
Includes pair, bond, and bending terms involving this monomer.
"""
function _monomer_energy(sys::BeadSpringPolymer{D,T}, idx::Int) where {D,T}
    m   = Int(sys.polymer_id[idx])
    k   = Int(sys.monomer_k[idx])
    off = sys.offsets[m]
    M   = sys.lengths[m]

    E = _pair_energy_of(sys, idx, m, k)

    # Bond terms (max 2)
    k > 1 && (E += sys.bond_potential(minimum_image_sq(sys.positions[idx], sys.positions[off+k-1], sys.L)))
    k < M && (E += sys.bond_potential(minimum_image_sq(sys.positions[idx], sys.positions[off+k+1], sys.L)))

    # Bending terms (max 3, centre / left-shifted / right-shifted)
    if !(sys.bending_potential isa NoBendingPotential)
        k > 1 && k < M   && (E += sys.bending_potential(_cos_angle(sys.positions[off+k-1], sys.positions[idx],     sys.positions[off+k+1], sys.L)))
        k > 2             && (E += sys.bending_potential(_cos_angle(sys.positions[off+k-2], sys.positions[off+k-1], sys.positions[idx],     sys.L)))
        k < M-1           && (E += sys.bending_potential(_cos_angle(sys.positions[idx],     sys.positions[off+k+1], sys.positions[off+k+2], sys.L)))
    end

    return E
end

# ── Pair energy of single monomer: NoCellList ──────────────────────────────

@inline function _pair_energy_of(sys::BeadSpringPolymer{D,T,TPair,TBond,TBend,NoCellList},
                                  idx::Int, m::Int, k::Int) where {D,T,TPair<:AbstractPairPotential,TBond<:AbstractBondPotential,TBend<:AbstractBendingPotential}
    E = zero(T)
    pos = sys.positions[idx]
    @inbounds for j in 1:length(sys.positions)
        j == idx && continue
        Int(sys.polymer_id[j]) == m && abs(Int(sys.monomer_k[j]) - k) == 1 && continue
        r_sq = minimum_image_sq(pos, sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

# ── Pair energy of single monomer: CellList ────────────────────────────────

@inline function _pair_energy_of(sys::BeadSpringPolymer{D,T,TPair,TBond,TBend,CellList{D,K}},
                                  idx::Int, m::Int, k::Int) where {D,T,TPair<:AbstractPairPotential,TBond<:AbstractBondPotential,TBend<:AbstractBendingPotential,K}
    cl     = sys.cell_list
    pos    = sys.positions[idx]
    pot    = sys.pair_potential
    rc_sq  = cl.rc_sq
    L      = sys.L
    own_ci = cl.particle_cell[idx]
    E      = zero(T)
    @inbounds for ci in cl.interaction_cells[own_ci]
        for j in cl.cells[ci]
            j == idx && continue
            r_sq = _sq_dist(pos, sys.positions[j], L)
            if r_sq < rc_sq
                Int(sys.polymer_id[j]) == m && abs(Int(sys.monomer_k[j]) - k) == 1 && continue
                E += pot(r_sq)
            end
        end
    end
    return E
end

# ── Pair energy for chain translate (no bond exclusion) ────────────────────

@inline function _pair_energy_of_all(sys::BeadSpringPolymer{D,T,TPair,TBond,TBend,NoCellList},
                                      idx::Int) where {D,T,TPair<:AbstractPairPotential,TBond<:AbstractBondPotential,TBend<:AbstractBendingPotential}
    E = zero(T)
    @inbounds for j in 1:length(sys.positions)
        j == idx && continue
        r_sq = minimum_image_sq(sys.positions[idx], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline function _pair_energy_of_all(sys::BeadSpringPolymer{D,T,TPair,TBond,TBend,CellList{D,K}},
                                      idx::Int) where {D,T,TPair<:AbstractPairPotential,TBond<:AbstractBondPotential,TBend<:AbstractBendingPotential,K}
    cl = sys.cell_list
    pos = sys.positions[idx]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    own_ci = cl.particle_cell[idx]
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[own_ci]
        if ci == own_ci
            for j in cl.cells[ci]
                j == idx && continue
                r_sq = _sq_dist(pos, sys.positions[j], L)
                r_sq < rc_sq && (E += pot(r_sq))
            end
        else
            for j in cl.cells[ci]
                r_sq = _sq_dist(pos, sys.positions[j], L)
                r_sq < rc_sq && (E += pot(r_sq))
            end
        end
    end
    return E
end
