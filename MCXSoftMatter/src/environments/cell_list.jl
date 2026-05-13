"""
    NoCellList

Sentinel: no cell list acceleration. All-pairs loops are used.
"""
struct NoCellList end

"""
    CellList{D,K}

D-dimensional linked-list cell list for O(1) per-particle energy evaluation.

The box is divided into cells of size >= r_cutoff per dimension.
Supports anisotropic boxes (different Lx, Ly, Lz → different cell counts per dimension).
Each cell knows its `K = 3^D` neighbor cells (precomputed once, stored inline as
`NTuple{K,Int}` — no heap pointer per cell). Particles are assigned to cells via a
linked list: `head[cell]` gives the first particle, `next[particle]` gives the next
particle in the same cell (0 = end of list). No per-cell heap allocations.
"""
struct CellList{D, K}
    cell_size::SVector{D,Float64}            # cell size per dimension
    inv_cell_size::SVector{D,Float64}        # 1 / cell_size per dimension
    rc_sq::Float64                           # cutoff^2 (for energy evaluation)
    nc::SVector{D,Int}                       # cells per dimension
    head::Vector{Int}                        # cell -> first particle index (0 = empty)
    next::Vector{Int}                        # particle -> next particle in same cell (0 = end)
    particle_cell::Vector{Int}               # particle -> cell index
    interaction_cells::Vector{NTuple{K,Int}} # cell -> interacting cell indices (self + neighbors)
end

function CellList{D}(N::Int, box::PeriodicBox{D}, r_cutoff::Real) where D
    nc = SVector{D,Int}(ntuple(d -> max(3, floor(Int, box.L[d] / r_cutoff)), Val(D)))
    cs = SVector{D,Float64}(ntuple(d -> Float64(box.L[d]) / nc[d], Val(D)))
    inv_cs = SVector{D,Float64}(ntuple(d -> nc[d] / Float64(box.L[d]), Val(D)))
    num_total = prod(nc)
    head = zeros(Int, num_total)
    next = zeros(Int, N)
    particle_cell = zeros(Int, N)
    int_cells = _build_interaction_cells(Val(D), nc)
    CellList{D, 3^D}(cs, inv_cs, Float64(r_cutoff)^2, nc, head, next, particle_cell, int_cells)
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
    idx = 1; stride = 1
    @inbounds for d in 1:D
        ci = min(floor(Int, pos[d] * cl.inv_cell_size[d]), cl.nc[d] - 1)
        idx += ci * stride
        stride *= cl.nc[d]
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
