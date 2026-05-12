mutable struct ParticleSystem{D, T<:AbstractFloat,
                               TPair<:AbstractPairPotential,
                               TMol<:AbstractMolecule,
                               TCache,
                               TCell,
                               TIdx<:Integer} <: AbstractSoftMatterSystem
    positions::Vector{SVector{D,T}}
    molecules::Vector{TMol}
    molecule_id::Vector{TIdx}
    monomer_k::Vector{TIdx}
    L::T
    pair_potential::TPair
    cache::TCache
    cell_list::TCell
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

@inline function _cos_angle(a::SVector{D,T}, b::SVector{D,T}, c::SVector{D,T}, L) where {D,T}
    ba = minimum_image_displacement(a, b, L)
    bc = minimum_image_displacement(c, b, L)
    return sum(ba .* bc) / (sqrt(sum(abs2, ba)) * sqrt(sum(abs2, bc)))
end

@inline function _local_pair_energy_no_excl(sys::ParticleSystem{D,T,TPair,TMol,TC,NoCellList},
                                             i::Int) where {D,T,TPair,TMol,TC}
    E = zero(T)
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline function _local_pair_energy_no_excl(sys::ParticleSystem{D,T,TPair,TMol,TC,CellList{D,K}},
                                             i::Int) where {D,T,TPair,TMol,TC,K}
    cl = sys.cell_list
    pos = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[cl.particle_cell[i]]
        j = cl.head[ci]
        while j != 0
            if j != i
                r_sq = _sq_dist(pos, sys.positions[j], L)
                r_sq < rc_sq && (E += pot(r_sq))
            end
            j = cl.next[j]
        end
    end
    return E
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TPair,TMol,TC,NoCellList},
                                             i::Int, new_pos::SVector{D,T}) where {D,T,TPair,TMol,TC}
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    L = sys.L
    dE = zero(T)
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        pos_j = sys.positions[j]
        dE += pot(minimum_image_sq(new_pos, pos_j, L)) -
              pot(minimum_image_sq(old_pos, pos_j, L))
    end
    return dE
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TPair,TMol,TC,CellList{D,K}},
                                             i::Int, new_pos::SVector{D,T}) where {D,T,TPair,TMol,TC,K}
    cl = sys.cell_list
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    old_ci = cl.particle_cell[i]
    new_ci = cell_index(cl, new_pos)
    dE = zero(T)
    if old_ci == new_ci
        @inbounds for ci in cl.interaction_cells[old_ci]
            j = cl.head[ci]
            while j != 0
                if j != i
                    pos_j = sys.positions[j]
                    r_old = _sq_dist(old_pos, pos_j, L)
                    r_new = _sq_dist(new_pos, pos_j, L)
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
                    r_sq = _sq_dist(old_pos, sys.positions[j], L)
                    r_sq < rc_sq && (dE -= pot(r_sq))
                end
                j = cl.next[j]
            end
        end
        @inbounds for ci in cl.interaction_cells[new_ci]
            j = cl.head[ci]
            while j != 0
                if j != i
                    r_sq = _sq_dist(new_pos, sys.positions[j], L)
                    r_sq < rc_sq && (dE += pot(r_sq))
                end
                j = cl.next[j]
            end
        end
    end
    return dE
end

# ── Segment pair energy change (external interactions only) ──────────────────

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TPair,TMol,TC,NoCellList},
                                      start::Int, M::Int,
                                      displacement::SVector{D,T}) where {D,T,TPair,TMol,TC}
    pot = sys.pair_potential
    L = sys.L
    last = start + M - 1
    dE = zero(T)
    @inbounds for j in 1:length(sys.positions)
        (start <= j <= last) && continue
        pos_j = sys.positions[j]
        for idx in start:last
            old_pos = sys.positions[idx]
            dE += pot(minimum_image_sq(old_pos + displacement, pos_j, L)) -
                  pot(minimum_image_sq(old_pos, pos_j, L))
        end
    end
    return dE
end

@inline function _pair_energy_change(sys::ParticleSystem{D,T,TPair,TMol,TC,CellList{D,K}},
                                      start::Int, M::Int,
                                      displacement::SVector{D,T}) where {D,T,TPair,TMol,TC,K}
    cl = sys.cell_list
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    last = start + M - 1
    dE = zero(T)
    @inbounds for idx in start:last
        old_pos = sys.positions[idx]
        new_pos = wrap_position(old_pos + displacement, L)
        for ci in cl.interaction_cells[cl.particle_cell[idx]]
            j = cl.head[ci]
            while j != 0
                if !(start <= j <= last)
                    r_sq = _sq_dist(old_pos, sys.positions[j], L)
                    r_sq < rc_sq && (dE -= pot(r_sq))
                end
                j = cl.next[j]
            end
        end
        for ci in cl.interaction_cells[cell_index(cl, new_pos)]
            j = cl.head[ci]
            while j != 0
                if !(start <= j <= last)
                    r_sq = _sq_dist(new_pos, sys.positions[j], L)
                    r_sq < rc_sq && (dE += pot(r_sq))
                end
                j = cl.next[j]
            end
        end
    end
    return dE
end
