mutable struct ParticleSystem{D, T<:AbstractFloat,
                               TEnv<:AbstractEnvironment{D,T},
                               TPair<:AbstractPairPotential,
                               TMol<:AbstractMolecule,
                               TCache,
                               TCell,
                               TIdx<:Integer} <: AbstractSoftMatterSystem
    env::TEnv
    positions::Vector{SVector{D,T}}
    molecules::Vector{TMol}
    molecule_id::Vector{TIdx}
    monomer_k::Vector{TIdx}
    pair_potential::TPair
    cache::TCache
    cell_list::TCell
end

function _make_cell_list(::Val{D}, N::Int, env::PeriodicBox{D}, pot::AbstractPairPotential) where D
    rc_sq = cutoff_sq(pot)
    if isfinite(rc_sq)
        rc = sqrt(rc_sq)
        nc_min = minimum(d -> floor(Int, env.L[d] / rc), 1:D)
        nc_min >= 3 || return NoCellList()
        return CellList{D}(N, env, rc)
    end
    return NoCellList()
end

@inline function _index_type(maxval::Int)
    maxval <= typemax(Int8)  && return Int8
    maxval <= typemax(Int16) && return Int16
    return Int32
end

num_particles(sys::ParticleSystem) = length(sys.positions)

function energy(sys::ParticleSystem; full::Bool=false)
    full && _recompute_energy!(sys)
    return total_energy(sys.cache)
end

energy_pair(sys::ParticleSystem) = _compute_pair_energy(sys)

@inline function _cos_angle(a::SVector{D,T}, b::SVector{D,T}, c::SVector{D,T}, env::AbstractEnvironment{D}) where {D,T}
    ba = difference(env, a, b)
    bc = difference(env, c, b)
    return sum(ba .* bc) / (sqrt(sum(abs2, ba)) * sqrt(sum(abs2, bc)))
end

@inline function _local_pair_energy_no_excl(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,NoCellList},
                                             i::Int) where {D,T,TEnv,TPair,TMol,TC}
    E = zero(T)
    env = sys.env
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        r_sq = distance_sq(env, sys.positions[i], sys.positions[j])
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline function _local_pair_energy_no_excl(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,CellList{D,K}},
                                             i::Int) where {D,T,TEnv,TPair,TMol,TC,K}
    cl = sys.cell_list
    pos = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    env = sys.env
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[cl.particle_cell[i]]
        j = cl.head[ci]
        while j != 0
            if j != i
                r_sq = distance_sq(env, pos, sys.positions[j])
                r_sq < rc_sq && (E += pot(r_sq))
            end
            j = cl.next[j]
        end
    end
    return E
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,NoCellList},
                                             i::Int, new_pos::SVector{D,T}) where {D,T,TEnv,TPair,TMol,TC}
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    env = sys.env
    dE = zero(T)
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        pos_j = sys.positions[j]
        dE += pot(distance_sq(env, new_pos, pos_j)) -
              pot(distance_sq(env, old_pos, pos_j))
    end
    return dE
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,CellList{D,K}},
                                             i::Int, new_pos::SVector{D,T}) where {D,T,TEnv,TPair,TMol,TC,K}
    cl = sys.cell_list
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    env = sys.env
    old_ci = cl.particle_cell[i]
    new_ci = cell_index(cl, new_pos)
    dE = zero(T)
    if old_ci == new_ci
        @inbounds for ci in cl.interaction_cells[old_ci]
            j = cl.head[ci]
            while j != 0
                if j != i
                    pos_j = sys.positions[j]
                    r_old = distance_sq(env, old_pos, pos_j)
                    r_new = distance_sq(env, new_pos, pos_j)
                    r_old < rc_sq && (dE -= pot(r_old))
                    r_new < rc_sq && (dE += pot(r_new))
                end
                j = cl.next[j]
            end
        end
    else
        @inbounds for ci in cl.interaction_cells[old_ci]
            j = cl.head[ci]
            while j != 0
                if j != i
                    r_sq = distance_sq(env, old_pos, sys.positions[j])
                    r_sq < rc_sq && (dE -= pot(r_sq))
                end
                j = cl.next[j]
            end
        end
        @inbounds for ci in cl.interaction_cells[new_ci]
            j = cl.head[ci]
            while j != 0
                if j != i
                    r_sq = distance_sq(env, new_pos, sys.positions[j])
                    r_sq < rc_sq && (dE += pot(r_sq))
                end
                j = cl.next[j]
            end
        end
    end
    return dE
end

# ── Segment pair energy change (external interactions only) ──────────────────

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,NoCellList},
                                      start::Int, M::Int,
                                      displacement::SVector{D,T}) where {D,T,TEnv,TPair,TMol,TC}
    pot = sys.pair_potential
    env = sys.env
    last = start + M - 1
    dE = zero(T)
    @inbounds for j in 1:length(sys.positions)
        (start <= j <= last) && continue
        pos_j = sys.positions[j]
        for idx in start:last
            old_pos = sys.positions[idx]
            dE += pot(distance_sq(env, old_pos + displacement, pos_j)) -
                  pot(distance_sq(env, old_pos, pos_j))
        end
    end
    return dE
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TEnv,TPair,TMol,TC,CellList{D,K}},
                                      start::Int, M::Int,
                                      displacement::SVector{D,T}) where {D,T,TEnv,TPair,TMol,TC,K}
    cl = sys.cell_list
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    env = sys.env
    last = start + M - 1
    dE = zero(T)
    @inbounds for idx in start:last
        old_pos = sys.positions[idx]
        new_pos = constrain(env, old_pos + displacement)
        for ci in cl.interaction_cells[cl.particle_cell[idx]]
            j = cl.head[ci]
            while j != 0
                if !(start <= j <= last)
                    r_sq = distance_sq(env, old_pos, sys.positions[j])
                    r_sq < rc_sq && (dE -= pot(r_sq))
                end
                j = cl.next[j]
            end
        end
        for ci in cl.interaction_cells[cell_index(cl, new_pos)]
            j = cl.head[ci]
            while j != 0
                if !(start <= j <= last)
                    r_sq = distance_sq(env, new_pos, sys.positions[j])
                    r_sq < rc_sq && (dE += pot(r_sq))
                end
                j = cl.next[j]
            end
        end
    end
    return dE
end
