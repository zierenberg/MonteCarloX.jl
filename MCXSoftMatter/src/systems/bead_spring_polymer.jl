"""
    BeadSpringPolymer(; D=3, num_poly, length_poly=nothing, lengths=nothing, L, ...)

Construct a `ParticleSystem` of bead-spring polymers.
`L` can be a scalar (cubic box) or an `SVector` (anisotropic box).
"""
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

    if L isa Real
        box = PeriodicBox{D}(L)
    else
        box = PeriodicBox(SVector{D}(float.(L)))
    end
    T = eltype(box.L)

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
    cl = _make_cell_list(Val(D), n_total, box, pair_potential)
    cache = CachePolymer(zero(T), zero(T), zero(T))
    ParticleSystem{D, T, typeof(pair_potential), typeof(molecules[1]),
                   typeof(cache), typeof(cl), TIdx}(
        positions, molecules, mol_id, mono_k, box,
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
    box = sys.box
    for mol in sys.molecules
        off = mol.offset
        M   = mol.length
        pos = SVector{D,T}(ntuple(d -> rand(rng, T) * box.L[d], Val(D)))
        sys.positions[off + 1] = pos
        for k in 2:M
            step = _random_unit_vector(Val(D), T, rng)
            pos  = wrap_position(pos + step, box)
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
    box = sys.box
    nmol = length(sys.molecules)
    @inbounds for mi in 1:nmol
        mol_i = sys.molecules[mi]
        off_i = mol_i.offset
        M_i   = mol_i.length
        for ki in 1:M_i-2
            i = off_i + ki
            for kj in ki+2:M_i
                j = off_i + kj
                r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], box)
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
                    r_sq = minimum_image_sq(sys.positions[i], sys.positions[j], box)
                    E += sys.pair_potential(r_sq)
                end
            end
        end
    end
    return E
end

function _compute_bond_energy(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    E = zero(T)
    box = sys.box
    @inbounds for mol in sys.molecules
        mol.bond isa NoBondPotential && continue
        off = mol.offset
        for k in 1:mol.length-1
            r_sq = minimum_image_sq(sys.positions[off+k], sys.positions[off+k+1], box)
            E += mol.bond(r_sq)
        end
    end
    return E
end

function _compute_bending_energy(sys::ParticleSystem{D,T,P,<:Polymer}) where {D,T,P}
    E = zero(T)
    box = sys.box
    @inbounds for mol in sys.molecules
        mol.bend isa NoBendingPotential && continue
        off = mol.offset
        for k in 1:mol.length-2
            cos_theta = _cos_angle(sys.positions[off+k], sys.positions[off+k+1], sys.positions[off+k+2], box)
            E += mol.bend(cos_theta)
        end
    end
    return E
end

@inline function _local_pair_energy(sys::ParticleSystem{D,T,TPair,<:Polymer,TC,NoCellList},
                                     i::Int) where {D,T,TPair,TC}
    E = zero(T)
    pos_i = sys.positions[i]
    box = sys.box
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1 && continue
        r_sq = minimum_image_sq(pos_i, sys.positions[j], box)
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
    box = sys.box
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    E = zero(T)
    @inbounds for ci in cl.interaction_cells[cl.particle_cell[i]]
        j = cl.head[ci]
        while j != 0
            if j != i
                r_sq = _sq_dist(pos_i, sys.positions[j], box)
                if r_sq < rc_sq
                    if !(Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1)
                        E += pot(r_sq)
                    end
                end
            end
            j = cl.next[j]
        end
    end
    return E
end

@inline function _pair_energy_change_excl(sys::ParticleSystem{D,T,TPair,<:Polymer,TC,NoCellList},
                                     i::Int, new_pos::SVector{D,T}) where {D,T,TPair,TC}
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    box = sys.box
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    dE = zero(T)
    @inbounds for j in 1:length(sys.positions)
        j == i && continue
        Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1 && continue
        pos_j = sys.positions[j]
        dE += pot(minimum_image_sq(new_pos, pos_j, box)) -
              pot(minimum_image_sq(old_pos, pos_j, box))
    end
    return dE
end

@inline function _pair_energy_change_excl(sys::ParticleSystem{D,T,TPair,<:Polymer,TC,CellList{D,K}},
                                     i::Int, new_pos::SVector{D,T}) where {D,T,TPair,TC,K}
    cl = sys.cell_list
    old_pos = sys.positions[i]
    pot = sys.pair_potential
    rc_sq = cl.rc_sq
    box = sys.box
    mol_id = sys.molecule_id
    mono_k = sys.monomer_k
    m = Int(mol_id[i])
    k = Int(mono_k[i])
    dE = zero(T)
    @inbounds for ci in cl.interaction_cells[cl.particle_cell[i]]
        j = cl.head[ci]
        while j != 0
            if j != i && !(Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1)
                r_sq = _sq_dist(old_pos, sys.positions[j], box)
                r_sq < rc_sq && (dE -= pot(r_sq))
            end
            j = cl.next[j]
        end
    end
    @inbounds for ci in cl.interaction_cells[cell_index(cl, new_pos)]
        j = cl.head[ci]
        while j != 0
            if j != i && !(Int(mol_id[j]) == m && abs(Int(mono_k[j]) - k) == 1)
                r_sq = _sq_dist(new_pos, sys.positions[j], box)
                r_sq < rc_sq && (dE += pot(r_sq))
            end
            j = cl.next[j]
        end
    end
    return dE
end

function _monomer_energy(sys::ParticleSystem{D,T,P,<:Polymer}, idx::Int) where {D,T,P}
    m   = Int(sys.molecule_id[idx])
    k   = Int(sys.monomer_k[idx])
    mol = sys.molecules[m]
    off = mol.offset
    M   = mol.length
    box = sys.box

    E = _local_pair_energy(sys, idx)

    k > 1 && (E += mol.bond(minimum_image_sq(sys.positions[idx], sys.positions[off+k-1], box)))
    k < M && (E += mol.bond(minimum_image_sq(sys.positions[idx], sys.positions[off+k+1], box)))

    if !(mol.bend isa NoBendingPotential)
        k > 1 && k < M && (E += mol.bend(_cos_angle(sys.positions[off+k-1], sys.positions[idx],     sys.positions[off+k+1], box)))
        k > 2           && (E += mol.bend(_cos_angle(sys.positions[off+k-2], sys.positions[off+k-1], sys.positions[idx],     box)))
        k < M-1         && (E += mol.bend(_cos_angle(sys.positions[idx],     sys.positions[off+k+1], sys.positions[off+k+2], box)))
    end

    return E
end

function _monomer_energy_change(sys::ParticleSystem{D,T,P,<:Polymer}, idx::Int, new_pos::SVector{D,T}) where {D,T,P}
    m   = Int(sys.molecule_id[idx])
    k   = Int(sys.monomer_k[idx])
    mol = sys.molecules[m]
    off = mol.offset
    M   = mol.length
    old_pos = sys.positions[idx]
    box = sys.box

    dE = _pair_energy_change_excl(sys, idx, new_pos)

    if k > 1
        nb = sys.positions[off+k-1]
        dE += mol.bond(minimum_image_sq(new_pos, nb, box)) -
              mol.bond(minimum_image_sq(old_pos, nb, box))
    end
    if k < M
        nb = sys.positions[off+k+1]
        dE += mol.bond(minimum_image_sq(new_pos, nb, box)) -
              mol.bond(minimum_image_sq(old_pos, nb, box))
    end

    if !(mol.bend isa NoBendingPotential)
        if k > 1 && k < M
            prev = sys.positions[off+k-1]
            next = sys.positions[off+k+1]
            dE += mol.bend(_cos_angle(prev, new_pos, next, box)) -
                  mol.bend(_cos_angle(prev, old_pos, next, box))
        end
        if k > 2
            pp = sys.positions[off+k-2]
            prev = sys.positions[off+k-1]
            dE += mol.bend(_cos_angle(pp, prev, new_pos, box)) -
                  mol.bend(_cos_angle(pp, prev, old_pos, box))
        end
        if k < M-1
            next  = sys.positions[off+k+1]
            next2 = sys.positions[off+k+2]
            dE += mol.bend(_cos_angle(new_pos, next, next2, box)) -
                  mol.bend(_cos_angle(old_pos, next, next2, box))
        end
    end

    return dE
end
