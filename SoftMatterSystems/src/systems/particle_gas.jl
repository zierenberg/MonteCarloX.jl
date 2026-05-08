"""
    ParticleGas{D, T, TPair, TCell} <: AbstractSoftMatterSystem

D-dimensional particle gas in a cubic box with periodic boundary conditions.

When the pair potential has a finite cutoff, a cell list is automatically
created for O(N) pair energy evaluation.

# Constructor
    ParticleGas(; D=3, N, L, pair_potential)
    ParticleGas(; D=3, N, rho, pair_potential)   # from density
"""
mutable struct ParticleGas{D, T<:AbstractFloat, TPair<:AbstractPairPotential, TCell} <: AbstractSoftMatterSystem
    positions::Vector{SVector{D,T}}
    N::Int
    L::T
    pair_potential::TPair
    cached_energy::T
    cell_list::TCell
end

# ── Constructors ─────────────────────────────────────────────────────────────

function ParticleGas(; D::Int=3,
                       N::Integer,
                       L=nothing,
                       rho=nothing,
                       pair_potential::AbstractPairPotential)
    @assert (L !== nothing) ⊻ (rho !== nothing) "Provide either `L` or `rho`, not both"
    if rho !== nothing
        L = (N / rho)^(1/D)
    end
    T = typeof(float(L))
    positions = [zero(SVector{D,T}) for _ in 1:N]
    cl = _make_cell_list(Val(D), Int(N), T(L), pair_potential)
    ParticleGas{D, T, typeof(pair_potential), typeof(cl)}(
        positions, Int(N), T(L), pair_potential, zero(T), cl)
end

function _make_cell_list(::Val{D}, N::Int, L, pot::AbstractPairPotential) where D
    rc_sq = cutoff_sq(pot)
    if isfinite(rc_sq)
        rc = sqrt(rc_sq)
        nc = floor(Int, L / rc)
        nc >= 3 || return NoCellList()
        return CellList{D}(N, L, rc)
    end
    return NoCellList()
end

# ── Accessors ────────────────────────────────────────────────────────────────

num_particles(sys::ParticleGas) = sys.N

# ── Initialization ───────────────────────────────────────────────────────────

function init!(sys::ParticleGas{D,T}, type::Symbol; rng=nothing) where {D,T}
    if type == :random
        @assert rng !== nothing "Random initialization requires rng"
        for i in 1:sys.N
            sys.positions[i] = SVector{D,T}(ntuple(_ -> rand(rng, T) * sys.L, Val(D)))
        end
    else
        error("Unknown initialization type: $type")
    end
    build!(sys.cell_list, sys.positions)
    _recompute_energy!(sys)
    return sys
end

# ── Energy ───────────────────────────────────────────────────────────────────

function _recompute_energy!(sys::ParticleGas{D,T}) where {D,T}
    E = zero(T)
    @inbounds for i in 1:sys.N-1
        for j in i+1:sys.N
            r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
            E += sys.pair_potential(r_sq)
        end
    end
    sys.cached_energy = E
    return nothing
end

function energy(sys::ParticleGas; full::Bool=false)
    full && _recompute_energy!(sys)
    return sys.cached_energy
end

energy_pair(sys::ParticleGas) = energy(sys; full=true)

# ── Per-particle energy: NoCellList (brute force) ──────────────────────────

@inline function _energy_of_particle(sys::ParticleGas{D,T,TPair,NoCellList}, i::Int) where {D,T,TPair<:AbstractPairPotential}
    E = zero(T)
    @inbounds for j in 1:sys.N
        j == i && continue
        r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

# ── Per-particle energy: CellList (O(1) via neighbor cells) ────────────────

@inline function _energy_of_particle(sys::ParticleGas{D,T,TPair,CellList{D,K}}, i::Int) where {D,T,TPair<:AbstractPairPotential,K}
    cl = sys.cell_list
    pos_i = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    own_ci = cl.particle_cell[i]
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[own_ci]
        if ci == own_ci
            for j in cl.cells[ci]
                j == i && continue
                r_sq = _sq_dist(pos_i, sys.positions[j], L)
                r_sq < rc_sq && (E += pot(r_sq))
            end
        else
            for j in cl.cells[ci]
            r_sq = _sq_dist(pos_i, sys.positions[j], L)
            r_sq < rc_sq && (E += pot(r_sq))
            end
        end
    end
    return E
end
