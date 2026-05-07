"""
    NoCellList

Sentinel: no cell list acceleration. All-pairs loops are used.
"""
struct NoCellList end

"""
    CellList{D}

D-dimensional cell list for O(1) per-particle energy evaluation.

The box is divided into cells of size >= r_cutoff. Each cell knows its
`3^D` neighbor cells (precomputed once). Particles are assigned to cells
and energy evaluation iterates only particles in neighbor cells.

Cell assignment is updated in-place when a particle moves — no rebuild needed.
"""
struct CellList{D}
    cell_size::Float64
    rc_sq::Float64                          # cutoff^2 (for energy evaluation)
    cells::Vector{Vector{Int}}              # cell -> particle indices
    particle_cell::Vector{Int}              # particle -> cell index
    neighbor_cells::Vector{Vector{Int}}     # cell -> neighbor cell indices
end

function CellList{D}(N::Int, L::Real, r_cutoff::Real) where D
    nc_per_dim = max(3, floor(Int, L / r_cutoff))
    cs = Float64(L) / nc_per_dim
    num_total = nc_per_dim^D
    cells = [Int[] for _ in 1:num_total]
    particle_cell = zeros(Int, N)
    nbr_cells = _build_neighbor_cells(Val(D), nc_per_dim)
    CellList{D}(cs, Float64(r_cutoff)^2, cells, particle_cell, nbr_cells)
end

# Number of cells per dimension
@inline _nc_per_dim(cl::CellList{D}) where D = round(Int, length(cl.cells)^(1/D))

# ── Precompute neighbor cell indices ────────────────────────────────────────

function _build_neighbor_cells(::Val{D}, nc::Int) where D
    offsets = _build_neighbor_offsets(Val(D))
    num_total = nc^D
    nbrs = Vector{Vector{Int}}(undef, num_total)
    for ci in 1:num_total
        nbrs[ci] = [_neighbor_cell_index(ci, offset, nc) for offset in offsets]
    end
    return nbrs
end

function _build_neighbor_offsets(::Val{D}) where D
    offsets = NTuple{D,Int}[]
    _enumerate_offsets!(offsets, Val(D), Int[])
    return offsets
end

function _enumerate_offsets!(offsets, ::Val{0}, current)
    push!(offsets, Tuple(current))
    return nothing
end

function _enumerate_offsets!(offsets, ::Val{R}, current) where R
    for d in -1:1
        push!(current, d)
        _enumerate_offsets!(offsets, Val(R-1), current)
        pop!(current)
    end
    return nothing
end

function _neighbor_cell_index(base::Int, offset::NTuple{D,Int}, nc::Int) where D
    idx = 1; stride = 1; rem = base - 1
    @inbounds for d in 1:D
        ci = rem % nc; rem = rem ÷ nc
        idx += mod(ci + offset[d], nc) * stride
        stride *= nc
    end
    return idx
end

# ── Cell index from position ────────────────────────────────────────────────

@inline function cell_index(cl::CellList{D}, pos::SVector{D}) where D
    nc = _nc_per_dim(cl)
    idx = 1; stride = 1
    @inbounds for d in 1:D
        ci = min(floor(Int, pos[d] / cl.cell_size), nc - 1)
        idx += ci * stride
        stride *= nc
    end
    return idx
end

# ── Assign all particles to cells ──────────────────────────────────────────

function build!(cl::CellList{D}, positions::AbstractVector{<:SVector{D}}) where D
    for c in cl.cells; empty!(c); end
    @inbounds for i in eachindex(positions)
        ci = cell_index(cl, positions[i])
        cl.particle_cell[i] = ci
        push!(cl.cells[ci], i)
    end
    return nothing
end

# ── Update cell assignment for a single particle ───────────────────────────

@inline function update_particle!(cl::CellList{D}, i::Int, new_pos::SVector{D}) where D
    new_ci = cell_index(cl, new_pos)
    old_ci = cl.particle_cell[i]
    if new_ci != old_ci
        _remove_from_list!(cl.cells[old_ci], i)
        push!(cl.cells[new_ci], i)
        cl.particle_cell[i] = new_ci
    end
    return nothing
end

@inline function _remove_from_list!(list::Vector{Int}, val::Int)
    @inbounds for k in eachindex(list)
        if list[k] == val
            list[k] = list[end]
            pop!(list)
            return nothing
        end
    end
    return nothing
end

# ── NoCellList dispatches (no-ops) ─────────────────────────────────────────

build!(::NoCellList, positions) = nothing
update_particle!(::NoCellList, i, new_pos) = nothing
