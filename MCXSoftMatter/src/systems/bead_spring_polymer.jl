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

    molecules = [Polymer(offs[m], lens[m], bond_potential, bending_potential) for m in 1:num_poly_i]

    TIdx = _index_type(max(num_poly_i, maximum(lens)))
    mol_id = Vector{TIdx}(undef, n_total)
    mono_k = Vector{TIdx}(undef, n_total)
    for m in 1:num_poly_i
        for k in 1:lens[m]
            i = offs[m] + k
            mol_id[i] = TIdx(m)
            mono_k[i] = TIdx(k)
        end
    end

    positions = [zero(SVector{D,T}) for _ in 1:n_total]
    cl = _make_cell_list(Val(D), n_total, T(L), pair_potential)
    cache = CachePolymer(zero(T), zero(T), zero(T))
    ParticleSystem{D, T, typeof(pair_potential), typeof(molecules[1]),
                   typeof(cache), typeof(cl), TIdx}(
        positions, molecules, mol_id, mono_k, T(L),
        pair_potential, cache, cl)
end

num_polymers(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P} = length(sys.molecules)
polymer_length(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P} = sys.molecules[1].length
polymer_length(sys::ParticleSystem{D,T,P,<:Polymer}, m::Int) where {D,T,P} = sys.molecules[m].length

function init!(sys::ParticleSystem{D,T,P,<:Polymer}, type::Symbol; rng=nothing) where {D,T,P}
    type == :random_walk || error("Unknown initialization type: $type")
    @assert rng !== nothing "Random walk initialization requires rng"
    _init_random_walk!(sys, rng)
    build!(sys.cell_list, sys.positions)
    _recompute_energy!(sys)
    return sys
end

function _init_random_walk!(sys::ParticleSystem{D,T}, rng) where {D,T}
    for mol in sys.molecules
        off = mol.offset
        M   = mol.length
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

function _recompute_energy!(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    sys.cache.pair = _compute_pair_energy(sys)
    sys.cache.bond = _compute_bond_energy(sys)
    sys.cache.bend = _compute_bending_energy(sys)
    return nothing
end

energy_bond(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P} = _compute_bond_energy(sys)
energy_bending(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P} = _compute_bending_energy(sys)

function _compute_pair_energy(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    E = zero(T)
    nmol = length(sys.molecules)
    @inbounds for mi in 1:nmol
        mol_i = sys.molecules[mi]
        off_i = mol_i.offset
        M_i   = mol_i.length
        for ki in 1:M_i-2
            i = off_i + ki
            for kj in ki+2:M_i
                j = off_i + kj
                r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                E += sys.pair_potential(r_sq)
            end
        end
        for mj in mi+1:nmol
            mol_j = sys.molecules[mj]
            off_j = mol_j.offset
            M_j   = mol_j.length
            for ki in 1:M_i
                i = off_i + ki
                for kj in 1:M_j
                    j = off_j + kj
                    r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], sys.L)
                    E += sys.pair_potential(r_sq)
                end
            end
        end
    end
    return E
end

function _compute_bond_energy(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    E = zero(T)
    @inbounds for mol in sys.molecules
        mol.bond isa NoBondPotential && continue
        off = mol.offset
        for k in 1:mol.length-1
            r_sq = minimum_image_sq(sys.positions[off+k], sys.positions[off+k+1], sys.L)
            E += mol.bond(r_sq)
        end
    end
    return E
end

function _compute_bending_energy(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    E = zero(T)
    @inbounds for mol in sys.molecules
        mol.bend isa NoBendingPotential && continue
        off = mol.offset
        for k in 1:mol.length-2
            cos_theta = _cos_angle(sys.positions[off+k], sys.positions[off+k+1], sys.positions[off+k+2], sys.L)
            E += mol.bend(cos_theta)
        end
    end
    return E
end

@inline function _local_pair_energy(sys::ParticleSystem{D,T,TPair,<:Polymer,TC,NoCellList},
                                     i::Int) where {D,T,TPair,TC}
    E = zero(T)
    pos_i = sys.positions[i]
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1 && continue
        r_sq = minimum_image_sq(pos_i, sys.positions[j], sys.L)
        E += sys.pair_potential(r_sq)
    end
    return E
end

@inline function _local_pair_energy(sys::ParticleSystem{D,T,TPair,<:Polymer,TC,CellList{D,K}},
                                     i::Int) where {D,T,TPair,TC,K}
    cl = sys.cell_list
    pos_i = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    L = sys.L
    own_ci = cl.particle_cell[i]
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[own_ci]
        for j in cl.cells[ci]
            j == i && continue
            r_sq = _sq_dist(pos_i, sys.positions[j], L)
            if r_sq < rc_sq
                Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1 && continue
                E += pot(r_sq)
            end
        end
    end
    return E
end

function _monomer_energy(sys::ParticleSystem{D,T,P,<:Polymer}, idx::Int) where {D,T,P}
    m   = Int(sys.molecule_id[idx])
    k   = Int(sys.monomer_k[idx])
    mol = sys.molecules[m]
    off = mol.offset
    M   = mol.length

    E = _local_pair_energy(sys, idx)

    k > 1 && (E += mol.bond(minimum_image_sq(sys.positions[idx], sys.positions[off+k-1], sys.L)))
    k < M && (E += mol.bond(minimum_image_sq(sys.positions[idx], sys.positions[off+k+1], sys.L)))

    if !(mol.bend isa NoBendingPotential)
        k > 1 && k < M && (E += mol.bend(_cos_angle(sys.positions[off+k-1], sys.positions[idx],     sys.positions[off+k+1], sys.L)))
        k > 2           && (E += mol.bend(_cos_angle(sys.positions[off+k-2], sys.positions[off+k-1], sys.positions[idx],     sys.L)))
        k < M-1         && (E += mol.bend(_cos_angle(sys.positions[idx],     sys.positions[off+k+1], sys.positions[off+k+2], sys.L)))
    end

    return E
end
