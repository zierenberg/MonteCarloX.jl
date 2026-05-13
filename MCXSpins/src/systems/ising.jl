abstract type AbstractIsing <: AbstractSpinSystem end

const _I_STATES = Int8[-1, 1]

# ── Shared observables ───────────────────────────────────────────────────────

@inline energy(sys::AbstractIsing; full=false) = full ? _full_energy(sys) : _cached_energy(sys)
@inline magnetization(sys::AbstractIsing; full=false) = full ? sum(sys.spins) : sys.cached_mag

# ── Shared initialization ────────────────────────────────────────────────────

function init!(sys::AbstractIsing, type::Symbol; rng=nothing)
    if type == :up
        sys.spins .= 1
    elseif type == :down
        sys.spins .= -1
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        sys.spins .= rand(rng, _I_STATES, length(sys.spins))
    else
        error("Unknown initialization type: $type")
    end
    _recompute_cached!(sys)
    return sys
end

# ── Interface: propose_state / delta_sys ─────────────────────────────────────

@inline propose_state(rng::AbstractRNG, sys::AbstractIsing, i) = Int8(-sys.spins[i])
@inline delta_sys(sys::AbstractIsing, i, s_new::Int8) = local_pair_interactions(sys, i)

# ═══════════════════════════════════════════════════════════════════════════════
# IsingLattice: D-dimensional periodic hypercubic lattice, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct IsingLattice{D, NN, TJ<:Real, TH} <: AbstractIsing
    spins::Vector{Int8}
    const nbrs::Vector{NTuple{NN,Int}}
    const J::TJ
    const h::TH
    cached_pair::Int        # unweighted half sum: Σ_{<i,j>} s_i s_j
    cached_mag::Int
    cached_field::Float64
end

function IsingLattice(dims::AbstractVector{<:Integer}; J::Real = 1,  h=0)
    if h isa AbstractVector
        @assert length(h) == prod(dims) "Field vector length must match number of sites"
        _ising_lattice(Tuple(Int.(dims)), J, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _ising_lattice(Tuple(Int.(dims)), J, float(h))
    else
        _ising_lattice(Tuple(Int.(dims)), J, NoField())
    end
end

function _ising_lattice(dims::NTuple{D,Int}, J, h) where D
    N = prod(dims)
    nbrs = _build_lattice_neighbors(dims)
    sys = IsingLattice{D, 2D, typeof(J), typeof(h)}(ones(Int8, N), nbrs, J, h, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::IsingLattice, i)
    @inbounds begin
        acc = 0
        for j in sys.nbrs[i]
            acc += sys.spins[j]
        end
        return acc
    end
end

@inline function local_pair_interactions(sys::IsingLattice, i)
    @inbounds return Int(sys.spins[i]) * neighbor_sum(sys, i)
end

@inline delta_energy(sys::IsingLattice, i) = delta_energy(sys, i, local_pair_interactions(sys, i))

@inline function delta_energy(sys::IsingLattice{D, NN, TJ, TH}, i, lpi) where {D, NN, TJ, TH<:Union{Real,AbstractVector}}
    @inbounds s = Int(sys.spins[i])
    return 2 * sys.J * lpi + 2 * _site_field(sys.h, i) * s
end

@inline delta_energy(sys::IsingLattice{D, NN, TJ, NoField}, i, lpi) where {D, NN, TJ} = 2 * sys.J * lpi

@inline function MonteCarloX.modify!(sys::IsingLattice{D, NN, TJ, TH}, i::Int, lpi) where {D, NN, TJ, TH}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2 * lpi
    sys.cached_mag -= 2 * s
    _update_field_cache!(sys, sys.h, i, -2.0 * s)
    return nothing
end

@inline function MonteCarloX.modify!(sys::IsingLattice{D, NN, TJ, NoField}, i::Int, lpi) where {D, NN, TJ}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2 * lpi
    sys.cached_mag -= 2 * s
    return nothing
end

@inline _cached_energy(sys::IsingLattice) =
    -float(sys.J) * sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field)

function _full_energy(sys::IsingLattice)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return float(-sys.J * (pair_full ÷ 2)) - _field_sum(sys.h, sys.spins)
end

function _recompute_cached!(sys::IsingLattice)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full ÷ 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# IsingGraph: arbitrary graph, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct IsingGraph{TJ<:Real, TH} <: AbstractIsing
    spins::Vector{Int8}
    const graph::SimpleGraph
    const nbrs::Vector{Vector{Int}}
    const J::TJ
    const h::TH
    cached_pair::Int
    cached_mag::Int
    cached_field::Float64
end

function IsingGraph(graph::SimpleGraph, J::Real; h=0)
    n = nv(graph)
    nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:n]
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _ising_graph(graph, nbrs, J, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _ising_graph(graph, nbrs, J, float(h))
    else
        _ising_graph(graph, nbrs, J, NoField())
    end
end

function _ising_graph(graph, nbrs, J, h)
    n = nv(graph)
    sys = IsingGraph{typeof(J), typeof(h)}(ones(Int8, n), graph, nbrs, J, h, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::IsingGraph, i)
    acc = 0
    @inbounds for j in sys.nbrs[i]
        acc += sys.spins[j]
    end
    return acc
end

@inline function local_pair_interactions(sys::IsingGraph, i)
    return Int(sys.spins[i]) * neighbor_sum(sys, i)
end

@inline delta_energy(sys::IsingGraph, i) = delta_energy(sys, i, local_pair_interactions(sys, i))
@inline function delta_energy(sys::IsingGraph{TJ, TH}, i, lpi) where {TJ, TH<:Union{Real,AbstractVector}}
    @inbounds s = Int(sys.spins[i])
    return 2 * sys.J * lpi + 2 * _site_field(sys.h, i) * s
end

@inline delta_energy(sys::IsingGraph{TJ, NoField}, i, lpi) where {TJ} = 2 * sys.J * lpi

@inline function MonteCarloX.modify!(sys::IsingGraph{TJ, TH}, i::Int, lpi) where {TJ, TH}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2 * lpi
    sys.cached_mag -= 2 * s
    _update_field_cache!(sys, sys.h, i, -2.0 * s)
    return nothing
end

@inline function MonteCarloX.modify!(sys::IsingGraph{TJ, NoField}, i::Int, lpi) where {TJ}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2 * lpi
    sys.cached_mag -= 2 * s
    return nothing
end

@inline _cached_energy(sys::IsingGraph) =
    -float(sys.J) * sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field)

function _full_energy(sys::IsingGraph)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return float(-sys.J * (pair_full ÷ 2)) - _field_sum(sys.h, sys.spins)
end

function _recompute_cached!(sys::IsingGraph)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full ÷ 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# IsingMatrix: sparse J_{ij}, arbitrary topology
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct IsingMatrix{TJ<:Real, TH} <: AbstractIsing
    spins::Vector{Int8}
    const J::SparseMatrixCSC{TJ,Int}
    const h::TH
    cached_pair::Float64    # J-weighted half sum
    cached_mag::Int
    cached_field::Float64
end

function IsingMatrix(J::SparseMatrixCSC{TJ,Int}; h=0) where {TJ<:Real}
    _check_symmetric(J)
    n = size(J, 1)
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _ising_matrix(J, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _ising_matrix(J, float(h))
    else
        _ising_matrix(J, NoField())
    end
end

function _ising_matrix(J, h)
    n = size(J, 1)
    sys = IsingMatrix{eltype(J), typeof(h)}(ones(Int8, n), J, h, 0.0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::IsingMatrix, i)
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i+1]-1)
        j = sys.J.rowval[ptr]
        j != i && (acc += sys.J.nzval[ptr] * sys.spins[j])
    end
    return acc
end

@inline function local_pair_interactions(sys::IsingMatrix, i)
    return sys.spins[i] * neighbor_sum(sys, i)
end

@inline delta_energy(sys::IsingMatrix, i) = delta_energy(sys, i, local_pair_interactions(sys, i))
@inline function delta_energy(sys::IsingMatrix{TJ, TH}, i, lpi) where {TJ, TH<:Union{Real,AbstractVector}}
    @inbounds s = Int(sys.spins[i])
    return 2 * lpi + 2 * _site_field(sys.h, i) * s
end

@inline delta_energy(sys::IsingMatrix{TJ, NoField}, i, lpi) where {TJ} = 2 * lpi

@inline function MonteCarloX.modify!(sys::IsingMatrix{TJ, TH}, i::Int, lpi) where {TJ, TH<:Union{Real,AbstractVector}}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2.0 * lpi
    sys.cached_mag -= 2 * s
    _update_field_cache!(sys, sys.h, i, -2.0 * s)
    return nothing
end

@inline function MonteCarloX.modify!(sys::IsingMatrix{TJ, NoField}, i::Int, lpi) where {TJ}
    @inbounds s = Int(sys.spins[i])
    sys.spins[i] = Int8(-s)
    sys.cached_pair -= 2.0 * lpi
    sys.cached_mag -= 2 * s
    return nothing
end

@inline _cached_energy(sys::IsingMatrix) =
    -sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field)

function _full_energy(sys::IsingMatrix)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return -(pair_full / 2) - _field_sum(sys.h, sys.spins)
end

function _recompute_cached!(sys::IsingMatrix)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full / 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Ising factory constructors
# ═══════════════════════════════════════════════════════════════════════════════

function Ising(dims::AbstractVector{<:Integer}; J=1, periodic=true, h=0)
    if J isa Real && periodic
        return IsingLattice(dims; J=J, h=h)
    end
    graph = Graphs.SimpleGraphs.grid(collect(Int, dims); periodic)
    if J isa Real
        return IsingGraph(graph, J; h=h)
    elseif J isa AbstractVector{<:Real}
        return Ising(graph, J; h=h)
    else
        error("J must be Real or AbstractVector{<:Real}")
    end
end

Ising(graph::SimpleGraph, J::Real; h=0) = IsingGraph(graph, J; h=h)
Ising(J::SparseMatrixCSC; h=0) = IsingMatrix(J; h=h)

function Ising(graph::SimpleGraph, J::AbstractVector{<:Real}; h=0)
    @assert ne(graph) == length(J) "Length of J vector must equal number of graph edges"
    TJ = float(eltype(J))
    rows = Vector{Int}(undef, 2ne(graph))
    cols = Vector{Int}(undef, 2ne(graph))
    vals = Vector{TJ}(undef, 2ne(graph))
    k = 1
    for (idx, e) in enumerate(edges(graph))
        i, j = src(e), dst(e)
        Jij = TJ(J[idx])
        rows[k] = i; cols[k] = j; vals[k] = Jij; k += 1
        rows[k] = j; cols[k] = i; vals[k] = Jij; k += 1
    end
    return IsingMatrix(sparse(rows, cols, vals, nv(graph), nv(graph)); h=h)
end
