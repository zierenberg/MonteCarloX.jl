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
    own_ci = cl.particle_cell[i]
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[own_ci]
        if ci == own_ci
            for j in cl.cells[ci]
                j == i && continue
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
