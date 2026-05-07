"""
    BeadSpringPolymer{D, T, TPair, TBond, TBend, TCell} <: AbstractSoftMatterSystem

D-dimensional bead-spring polymer system in a cubic box with PBC.

The interaction is decomposed into composable potentials:
- `pair_potential` -- non-bonded pair interaction (e.g. LJ)
- `bond_potential` -- covalent bond along backbone (e.g. FENE)
- `bending_potential` -- bending stiffness at chain angles (e.g. cosine)

Positions are stored flat in polymer-major order: monomers 1..N of polymer 1,
then monomers 1..N of polymer 2, etc.

When the pair potential has a finite cutoff, a cell list is automatically
created for O(N) pair energy evaluation.

# Constructor
    BeadSpringPolymer(; D=3, num_poly, length_poly, L, pair_potential, bond_potential,
                        bending_potential=NoBendingPotential(), delta=0.1)
"""
mutable struct BeadSpringPolymer{D, T<:AbstractFloat,
                                  TPair<:AbstractPairPotential,
                                  TBond<:AbstractBondPotential,
                                  TBend<:AbstractBendingPotential,
                                  TCell} <: AbstractSoftMatterSystem
    positions::Vector{SVector{D,T}}
    num_poly::Int
    length_poly::Int
    L::T
    pair_potential::TPair
    bond_potential::TBond
    bending_potential::TBend
    delta::T
    cached_energy::T
    cell_list::TCell
end

# ── Constructor ──────────────────────────────────────────────────────────────

function BeadSpringPolymer(; D::Int=3,
                             num_poly::Integer,
                             length_poly::Integer,
                             L,
                             pair_potential::AbstractPairPotential,
                             bond_potential::AbstractBondPotential,
                             bending_potential::AbstractBendingPotential=NoBendingPotential(),
                             delta=0.1)
    T = typeof(float(L))
    n_total = Int(num_poly) * Int(length_poly)
    positions = [zero(SVector{D,T}) for _ in 1:n_total]
    cl = _make_cell_list(Val(D), n_total, T(L), pair_potential)
    BeadSpringPolymer{D, T, typeof(pair_potential), typeof(bond_potential),
                      typeof(bending_potential), typeof(cl)}(
        positions, Int(num_poly), Int(length_poly), T(L),
        pair_potential, bond_potential, bending_potential,
        T(delta), zero(T), cl)
end

# ── Accessors ────────────────────────────────────────────────────────────────

num_polymers(sys::BeadSpringPolymer) = sys.num_poly
polymer_length(sys::BeadSpringPolymer) = sys.length_poly

# Index of k-th monomer (1-based) of polymer m (1-based)
@inline _monomer_idx(m, k, N) = (m - 1) * N + k

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
        pos = SVector{D,T}(ntuple(_ -> rand(rng, T) * sys.L, Val(D)))
        sys.positions[_monomer_idx(m, 1, sys.length_poly)] = pos
        for k in 2:sys.length_poly
            step = _random_unit_vector(Val(D), T, rng)
            pos = wrap_position(pos + step, sys.L)
            sys.positions[_monomer_idx(m, k, sys.length_poly)] = pos
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
        # Intra-polymer: skip covalent neighbors (ki, ki+1)
        for ki in 1:sys.length_poly-2
            i = _monomer_idx(m, ki, sys.length_poly)
            for kj in ki+2:sys.length_poly
                j = _monomer_idx(m, kj, sys.length_poly)
                r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                E += sys.pair_potential(r_sq)
            end
        end
        # Inter-polymer
        for m2 in m+1:sys.num_poly
            for ki in 1:sys.length_poly
                i = _monomer_idx(m, ki, sys.length_poly)
                for kj in 1:sys.length_poly
                    j = _monomer_idx(m2, kj, sys.length_poly)
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
        for k in 1:sys.length_poly-1
            i = _monomer_idx(m, k, sys.length_poly)
            j = _monomer_idx(m, k+1, sys.length_poly)
            r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
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
        for k in 1:sys.length_poly-2
            i = _monomer_idx(m, k, sys.length_poly)
            j = _monomer_idx(m, k+1, sys.length_poly)
            l = _monomer_idx(m, k+2, sys.length_poly)
            cos_theta = _cos_angle(sys.positions[i], sys.positions[j], sys.positions[l], sys.L)
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
    m = (idx - 1) ÷ sys.length_poly + 1   # polymer index
    k = (idx - 1) % sys.length_poly + 1   # monomer index within polymer

    E = _pair_energy_of(sys, idx, m, k)

    # Bond with predecessor
    if k > 1
        j = _monomer_idx(m, k-1, sys.length_poly)
        r_sq = minimum_image_sq(sys.positions[idx], sys.positions[j], sys.L)
        E += sys.bond_potential(r_sq)
    end

    # Bond with successor
    if k < sys.length_poly
        j = _monomer_idx(m, k+1, sys.length_poly)
        r_sq = minimum_image_sq(sys.positions[idx], sys.positions[j], sys.L)
        E += sys.bond_potential(r_sq)
    end

    # Bending contributions
    if !(sys.bending_potential isa NoBendingPotential)
        if k > 1 && k < sys.length_poly
            cos_theta = _cos_angle(
                sys.positions[_monomer_idx(m, k-1, sys.length_poly)],
                sys.positions[idx],
                sys.positions[_monomer_idx(m, k+1, sys.length_poly)], sys.L)
            E += sys.bending_potential(cos_theta)
        end
        if k > 2
            cos_theta = _cos_angle(
                sys.positions[_monomer_idx(m, k-2, sys.length_poly)],
                sys.positions[_monomer_idx(m, k-1, sys.length_poly)],
                sys.positions[idx], sys.L)
            E += sys.bending_potential(cos_theta)
        end
        if k < sys.length_poly - 1
            cos_theta = _cos_angle(
                sys.positions[idx],
                sys.positions[_monomer_idx(m, k+1, sys.length_poly)],
                sys.positions[_monomer_idx(m, k+2, sys.length_poly)], sys.L)
            E += sys.bending_potential(cos_theta)
        end
    end

    return E
end

# ── Pair energy of single monomer: NoCellList ──────────────────────────────

@inline function _pair_energy_of(sys::BeadSpringPolymer{D,T,<:Any,<:Any,<:Any,NoCellList},
                                  idx::Int, m::Int, k::Int) where {D,T}
    E = zero(T)
    pos = sys.positions[idx]
    n_total = sys.num_poly * sys.length_poly
    @inbounds for j in 1:n_total
        j == idx && continue
        mj = (j - 1) ÷ sys.length_poly + 1
        kj = (j - 1) % sys.length_poly + 1
        mj == m && abs(kj - k) == 1 && continue
        r_sq = minimum_image_sq(pos, sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

# ── Pair energy of single monomer: CellList ────────────────────────────────

@inline function _pair_energy_of(sys::BeadSpringPolymer{D,T,<:Any,<:Any,<:Any,CellList{D}},
                                  idx::Int, m::Int, k::Int) where {D,T}
    cl = sys.cell_list
    pos = sys.positions[idx]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    E = zero(T)
    @inbounds for nci in cl.neighbor_cells[cl.particle_cell[idx]]
        for j in cl.cells[nci]
            j == idx && continue
            r_sq = _sq_dist(pos, sys.positions[j], L)
            if r_sq < rc_sq
                mj = (j - 1) ÷ sys.length_poly + 1
                kj = (j - 1) % sys.length_poly + 1
                mj == m && abs(kj - k) == 1 && continue
                E += pot(r_sq)
            end
        end
    end
    return E
end

# ── Pair energy for chain translate (no bond exclusion) ────────────────────

@inline function _pair_energy_of_all(sys::BeadSpringPolymer{D,T,<:Any,<:Any,<:Any,NoCellList},
                                      idx::Int) where {D,T}
    E = zero(T)
    n_total = sys.num_poly * sys.length_poly
    @inbounds for j in 1:n_total
        j == idx && continue
        r_sq = minimum_image_sq(sys.positions[idx], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline function _pair_energy_of_all(sys::BeadSpringPolymer{D,T,<:Any,<:Any,<:Any,CellList{D}},
                                      idx::Int) where {D,T}
    cl = sys.cell_list
    pos = sys.positions[idx]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    E = zero(T)
    @inbounds for nci in cl.neighbor_cells[cl.particle_cell[idx]]
        for j in cl.cells[nci]
            j == idx && continue
            r_sq = _sq_dist(pos, sys.positions[j], L)
            r_sq < rc_sq && (E += pot(r_sq))
        end
    end
    return E
end
