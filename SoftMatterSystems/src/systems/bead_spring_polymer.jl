"""
    BeadSpringPolymer{T, TPair, TBond, TBend} <: AbstractSoftMatterSystem

Off-lattice bead-spring polymer system: M chains of N monomers each in a
cubic box with periodic boundary conditions.

The interaction is decomposed into three composable potentials:
- `pair_potential::TPair` -- non-bonded pair interaction (e.g. LJ, WCA)
- `bond_potential::TBond` -- covalent bond along chain backbone (e.g. FENE)
- `bending_potential::TBend` -- bending stiffness at chain angles (e.g. cosine)

Use `NoPotential` / `NoBondPotential` / `NoBendingPotential` to disable
any interaction. For example, a fully flexible chain has
`bending_potential = NoBendingPotential()`.

# Fields
- `positions::Vector{SVector{3,T}}` -- all monomer positions (M*N entries, polymer-major)
- `M::Int` -- number of polymers
- `N::Int` -- monomers per polymer
- `L::T` -- box side length
- `pair_potential::TPair`
- `bond_potential::TBond`
- `bending_potential::TBend`
- `delta::T` -- max displacement per component
- `cached_energy_pair::T`
- `cached_energy_bond::T`
- `cached_energy_bending::T`
"""
mutable struct BeadSpringPolymer{T<:AbstractFloat,
                                  TPair<:AbstractPairPotential,
                                  TBond<:AbstractBondPotential,
                                  TBend<:AbstractBendingPotential} <: AbstractSoftMatterSystem
    positions::Vector{SVector{3,T}}
    M::Int
    N::Int
    L::T
    pair_potential::TPair
    bond_potential::TBond
    bending_potential::TBend
    delta::T
    cached_energy_pair::T
    cached_energy_bond::T
    cached_energy_bending::T
end

"""
    BeadSpringPolymer(M, N; L, pair_potential, bond_potential, bending_potential=NoBendingPotential(), delta=0.1)

Construct an uninitialized bead-spring polymer system. Call `init!` to place monomers.
"""
function BeadSpringPolymer(M::Int, N::Int;
                            L::T,
                            pair_potential::TPair,
                            bond_potential::TBond,
                            bending_potential::TBend=NoBendingPotential(),
                            delta::T=T(0.1)
                            ) where {T<:AbstractFloat,
                                     TPair<:AbstractPairPotential,
                                     TBond<:AbstractBondPotential,
                                     TBend<:AbstractBendingPotential}
    positions = [zero(SVector{3,T}) for _ in 1:M*N]
    BeadSpringPolymer{T,TPair,TBond,TBend}(
        positions, M, N, L,
        pair_potential, bond_potential, bending_potential,
        delta, zero(T), zero(T), zero(T)
    )
end

# Index of k-th monomer (1-based) of polymer m (1-based)
@inline _monomer_idx(m, k, N) = (m - 1) * N + k

"""
    init!(sys::BeadSpringPolymer, type::Symbol; rng=nothing)

Initialize monomer positions.

- `:random_walk` -- place polymers as random walks with bond length 1.0
"""
function init!(sys::BeadSpringPolymer{T}, type::Symbol; rng=nothing) where T
    if type == :random_walk
        @assert rng !== nothing "Random walk initialization requires rng"
        for m in 1:sys.M
            # Random starting position
            pos = SVector{3,T}(rand(rng, T)*sys.L, rand(rng, T)*sys.L, rand(rng, T)*sys.L)
            sys.positions[_monomer_idx(m, 1, sys.N)] = pos
            for k in 2:sys.N
                # Random direction on unit sphere, step size 1.0
                theta = acos(T(2) * rand(rng, T) - T(1))
                phi = T(2π) * rand(rng, T)
                step = SVector{3,T}(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
                pos = wrap_position(pos + step, sys.L)
                sys.positions[_monomer_idx(m, k, sys.N)] = pos
            end
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_energy!(sys)
    return sys
end

function _recompute_energy!(sys::BeadSpringPolymer{T}) where T
    sys.cached_energy_pair = _compute_pair_energy(sys)
    sys.cached_energy_bond = _compute_bond_energy(sys)
    sys.cached_energy_bending = _compute_bending_energy(sys)
    return nothing
end

function _compute_pair_energy(sys::BeadSpringPolymer{T}) where T
    E = zero(T)
    n_total = sys.M * sys.N
    for m in 1:sys.M
        # Intra-polymer: skip covalent neighbors (i, i+1)
        for ki in 1:sys.N-2
            i = _monomer_idx(m, ki, sys.N)
            for kj in ki+2:sys.N
                j = _monomer_idx(m, kj, sys.N)
                r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                E += sys.pair_potential(r_sq)
            end
        end
        # Inter-polymer
        for m2 in m+1:sys.M
            for ki in 1:sys.N
                i = _monomer_idx(m, ki, sys.N)
                for kj in 1:sys.N
                    j = _monomer_idx(m2, kj, sys.N)
                    r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                    E += sys.pair_potential(r_sq)
                end
            end
        end
    end
    return E
end

function _compute_bond_energy(sys::BeadSpringPolymer{T}) where T
    sys.bond_potential isa NoBondPotential && return zero(T)
    E = zero(T)
    for m in 1:sys.M
        for k in 1:sys.N-1
            i = _monomer_idx(m, k, sys.N)
            j = _monomer_idx(m, k+1, sys.N)
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
@inline function _cos_angle(a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T}, L) where T
    ba = minimum_image_displacement(a, b, L)
    bc = minimum_image_displacement(c, b, L)
    dot_val = ba[1]*bc[1] + ba[2]*bc[2] + ba[3]*bc[3]
    norm_ba = sqrt(ba[1]^2 + ba[2]^2 + ba[3]^2)
    norm_bc = sqrt(bc[1]^2 + bc[2]^2 + bc[3]^2)
    return dot_val / (norm_ba * norm_bc)
end

function _compute_bending_energy(sys::BeadSpringPolymer{T}) where T
    sys.bending_potential isa NoBendingPotential && return zero(T)
    E = zero(T)
    for m in 1:sys.M
        for k in 1:sys.N-2
            i = _monomer_idx(m, k, sys.N)
            j = _monomer_idx(m, k+1, sys.N)
            l = _monomer_idx(m, k+2, sys.N)
            cos_theta = _cos_angle(sys.positions[i], sys.positions[j], sys.positions[l], sys.L)
            E += sys.bending_potential(cos_theta)
        end
    end
    return E
end

@inline function energy(sys::BeadSpringPolymer; full=false)
    full && _recompute_energy!(sys)
    return sys.cached_energy_pair + sys.cached_energy_bond + sys.cached_energy_bending
end

@inline energy_pair(sys::BeadSpringPolymer) = sys.cached_energy_pair
@inline energy_bond(sys::BeadSpringPolymer) = sys.cached_energy_bond
@inline energy_bending(sys::BeadSpringPolymer) = sys.cached_energy_bending
