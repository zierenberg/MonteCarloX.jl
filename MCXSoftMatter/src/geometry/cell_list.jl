"""
    NoCellList

Sentinel: no cell list acceleration. All-pairs loops are used.
"""
struct NoCellList end

"""
    CellList{D,K}

D-dimensional linked-list cell list for O(1) per-particle energy evaluation.

The box is divided into cells of size >= r_cutoff. Each cell knows its
`K = 3^D` neighbor cells (precomputed once, stored inline as `NTuple{K,Int}`
— no heap pointer per cell). Particles are assigned to cells via a linked list:
`head[cell]` gives the first particle, `next[particle]` gives the next particle
in the same cell (0 = end of list). No per-cell heap allocations.
"""
struct CellList{D, K}
    cell_size::Float64
    rc_sq::Float64                          # cutoff^2 (for energy evaluation)
    nc_per_dim::Int                         # cells per dimension (stored for hot path)
    head::Vector{Int}                       # cell -> first particle index (0 = empty)
    next::Vector{Int}                       # particle -> next particle in same cell (0 = end)
    particle_cell::Vector{Int}              # particle -> cell index
    interaction_cells::Vector{NTuple{K,Int}} # cell -> interacting cell indices (self + neighbors)
end

function CellList{D}(N::Int, L::Real, r_cutoff::Real) where D
    nc_per_dim = max(3, floor(Int, L / r_cutoff))
    cs = Float64(L) / nc_per_dim
    num_total = nc_per_dim^D
    head = zeros(Int, num_total)
    next = zeros(Int, N)
    particle_cell = zeros(Int, N)
    int_cells = _build_interaction_cells(Val(D), SVector{D,Int}(ntuple(_ -> nc_per_dim, Val(D))))
    CellList{D, 3^D}(cs, Float64(r_cutoff)^2, nc_per_dim, head, next, particle_cell, int_cells)
end

# ── Precompute interaction cell indices ──────────────────────────────────────

function _build_interaction_cells(::Val{D}, dims::SVector{D,Int}) where D
    N = prod(dims)
    strides = ntuple(d -> d == 1 ? 1 : prod(Tuple(dims)[1:d-1]), Val(D))
    int_cells = Vector{NTuple{3^D, Int}}(undef, N)
    for ci in 1:N
        c0 = ci - 1
        coords = ntuple(d -> (c0 ÷ strides[d]) % dims[d], Val(D))
        int_cells[ci] = ntuple(Val(3^D)) do k
            offset = ntuple(Val(D)) do d
                ((k - 1) ÷ (3^(d-1))) % 3 - 1
            end
            ci + sum(ntuple(d -> (mod(coords[d] + offset[d], dims[d]) - coords[d]) * strides[d], Val(D)))
        end
    end
    return int_cells
end

# ── Cell index from position ────────────────────────────────────────────────

@inline function cell_index(cl::CellList{D}, pos::SVector{D}) where D
    nc = cl.nc_per_dim
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
    fill!(cl.head, 0)
    @inbounds for i in eachindex(positions)
        ci = cell_index(cl, positions[i])
        cl.particle_cell[i] = ci
        cl.next[i] = cl.head[ci]
        cl.head[ci] = i
    end
    return nothing
end

# ── Update cell assignment for a single particle ───────────────────────────

@inline function update_particle!(cl::CellList{D}, i::Int, new_pos::SVector{D}) where D
    new_ci = cell_index(cl, new_pos)
    old_ci = cl.particle_cell[i]
    if new_ci != old_ci
        _remove_from_cell!(cl, i, old_ci)
        cl.next[i] = cl.head[new_ci]
        cl.head[new_ci] = i
        cl.particle_cell[i] = new_ci
    end
    return nothing
end

@inline function _remove_from_cell!(cl::CellList, i::Int, ci::Int)
    @inbounds if cl.head[ci] == i
        cl.head[ci] = cl.next[i]
    else
        prev = cl.head[ci]
        while cl.next[prev] != i
            prev = cl.next[prev]
        end
        cl.next[prev] = cl.next[i]
    end
    cl.next[i] = 0
    return nothing
end

# ── NoCellList dispatches (no-ops) ─────────────────────────────────────────

build!(::NoCellList, positions) = nothing
update_particle!(::NoCellList, i, new_pos) = nothing
