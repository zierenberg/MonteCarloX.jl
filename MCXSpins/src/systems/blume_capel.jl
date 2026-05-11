abstract type AbstractBlumeCapel <: AbstractSpinSystem end

const _BC_STATES = Int8[-1, 0, 1]
const BlumeCapelDelta{T} = NamedTuple{(:s_new, :delta_spin, :delta_spin2, :coupling), Tuple{Int8, Int, Int, T}}

# ── Shared observables ───────────────────────────────────────────────────────

@inline energy(sys::AbstractBlumeCapel; full=false) = full ? _full_energy(sys) : _cached_energy(sys)
@inline magnetization(sys::AbstractBlumeCapel; full=false) = full ? sum(sys.spins) : sys.cached_mag

# ── Shared initialization ────────────────────────────────────────────────────

function init!(sys::AbstractBlumeCapel, type::Symbol; rng=nothing)
    if type == :up
        sys.spins .= 1
    elseif type == :down
        sys.spins .= -1
    elseif type == :zero
        sys.spins .= 0
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        @inbounds for i in eachindex(sys.spins)
            sys.spins[i] = rand(rng, _BC_STATES)
        end
    else
        error("Unknown initialization type: $type")
    end
    _recompute_cached!(sys)
    return sys
end

# ── Interface: propose_state / delta_sys ─────────────────────────────────────

@inline function propose_state(rng::AbstractRNG, sys::AbstractBlumeCapel, i)
    @inbounds s_old = sys.spins[i]
    u = rand(rng, Bool)
    if s_old == Int8(-1)
        return u ? Int8(0) : Int8(1)
    elseif s_old == Int8(0)
        return u ? Int8(-1) : Int8(1)
    else
        return u ? Int8(-1) : Int8(0)
    end
end

@inline function delta_sys(sys::AbstractBlumeCapel, i, s_new::Int8)
    @inbounds s_old = Int(sys.spins[i])
    delta_spin = Int(s_new) - s_old
    delta_spin2 = Int(s_new)^2 - s_old^2
    coupling = neighbor_sum(sys, i)
    return (s_new=s_new, delta_spin=delta_spin, delta_spin2=delta_spin2, coupling=coupling)
end

@inline function delta_energy(sys::AbstractBlumeCapel, i, s_new::Int8)
    return delta_energy(sys, i, delta_sys(sys, i, s_new))
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelLattice: D-dimensional periodic hypercubic lattice, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelLattice{D, NN, TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const nbrs::Vector{NTuple{NN,Int}}
    const J::TJ
    const Δ::TC
    const h::TH
    cached_pair::Int        # unweighted half sum: Σ_{<i,j>} s_i s_j
    cached_mag::Int
    cached_spin2::Int       # Σ s_i²
    cached_field::Float64
end

function BlumeCapelLattice(dims::AbstractVector{<:Integer}, J::Real, Δ::Real; h=0)
    N = prod(dims)
    if h isa AbstractVector
        @assert length(h) == N "Field vector length must match number of sites"
        _bc_lattice(Tuple(Int.(dims)), J, Δ, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_lattice(Tuple(Int.(dims)), J, Δ, float(h))
    else
        _bc_lattice(Tuple(Int.(dims)), J, Δ, NoField())
    end
end

function _bc_lattice(dims::NTuple{D,Int}, J, Δ, h) where D
    N = prod(dims)
    nbrs = _build_lattice_neighbors(dims)
    sys = BlumeCapelLattice{D, 2D, typeof(J), typeof(Δ), typeof(h)}(
        ones(Int8, N), nbrs, J, Δ, h, 0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::BlumeCapelLattice{D}, i) where D
    @inbounds begin
        acc = 0
        for j in sys.nbrs[i]
            acc += sys.spins[j]
        end
        return acc
    end
end

@inline function local_pair_interactions(sys::BlumeCapelLattice{D}, i) where D
    @inbounds return Int(sys.spins[i]) * neighbor_sum(sys, i)
end

@inline function delta_energy(sys::BlumeCapelLattice{D, NN, TJ, TC, TH}, i, dsys::BlumeCapelDelta{Int}) where {D, NN, TJ, TC, TH<:Union{Real,AbstractVector}}
    local_term = sys.J * dsys.coupling + _site_field(sys.h, i)
    return -local_term * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function delta_energy(sys::BlumeCapelLattice{D, NN, TJ, TC, NoField}, i, dsys::BlumeCapelDelta{Int}) where {D, NN, TJ, TC}
    local_term = sys.J * dsys.coupling
    return -local_term * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function MonteCarloX.modify!(sys::BlumeCapelLattice{D, NN, TJ, TC, TH}, i::Int, dsys::BlumeCapelDelta{Int}) where {D, NN, TJ, TC, TH}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    _update_field_cache!(sys, sys.h, i, Float64(dsys.delta_spin))
    return nothing
end

@inline function MonteCarloX.modify!(sys::BlumeCapelLattice{D, NN, TJ, TC, NoField}, i::Int, dsys::BlumeCapelDelta{Int}) where {D, NN, TJ, TC}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    return nothing
end

@inline _cached_energy(sys::BlumeCapelLattice) =
    -float(sys.J) * sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field) +
    float(sys.Δ) * sys.cached_spin2

function _full_energy(sys::BlumeCapelLattice)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return float(-sys.J * (pair_full ÷ 2)) - _field_sum(sys.h, sys.spins) +
           float(sys.Δ) * sum(s -> Int(s)^2, sys.spins)
end

function _recompute_cached!(sys::BlumeCapelLattice)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full ÷ 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_spin2 = sum(s -> Int(s)^2, sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelGraph: arbitrary graph, uniform J
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelGraph{TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const graph::SimpleGraph
    const nbrs::Vector{Vector{Int}}
    const J::TJ
    const Δ::TC
    const h::TH
    cached_pair::Int
    cached_mag::Int
    cached_spin2::Int
    cached_field::Float64
end

function BlumeCapelGraph(graph::SimpleGraph, J::Real, Δ::Real; h=0)
    n = nv(graph)
    nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:n]
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _bc_graph(graph, nbrs, J, Δ, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_graph(graph, nbrs, J, Δ, float(h))
    else
        _bc_graph(graph, nbrs, J, Δ, NoField())
    end
end

function _bc_graph(graph, nbrs, J, Δ, h)
    n = nv(graph)
    sys = BlumeCapelGraph{typeof(J), typeof(Δ), typeof(h)}(
        ones(Int8, n), graph, nbrs, J, Δ, h, 0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::BlumeCapelGraph, i)
    acc = 0
    @inbounds for j in sys.nbrs[i]
        acc += sys.spins[j]
    end
    return acc
end

@inline function local_pair_interactions(sys::BlumeCapelGraph, i)
    return Int(sys.spins[i]) * neighbor_sum(sys, i)
end

@inline function delta_energy(sys::BlumeCapelGraph{TJ, TC, TH}, i, dsys::BlumeCapelDelta{Int}) where {TJ, TC, TH<:Union{Real,AbstractVector}}
    local_term = sys.J * dsys.coupling + _site_field(sys.h, i)
    return -local_term * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function delta_energy(sys::BlumeCapelGraph{TJ, TC, NoField}, i, dsys::BlumeCapelDelta{Int}) where {TJ, TC}
    local_term = sys.J * dsys.coupling
    return -local_term * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function MonteCarloX.modify!(sys::BlumeCapelGraph{TJ, TC, TH}, i::Int, dsys::BlumeCapelDelta{Int}) where {TJ, TC, TH}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    _update_field_cache!(sys, sys.h, i, Float64(dsys.delta_spin))
    return nothing
end

@inline function MonteCarloX.modify!(sys::BlumeCapelGraph{TJ, TC, NoField}, i::Int, dsys::BlumeCapelDelta{Int}) where {TJ, TC}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    return nothing
end

@inline _cached_energy(sys::BlumeCapelGraph) =
    -float(sys.J) * sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field) +
    float(sys.Δ) * sys.cached_spin2

function _full_energy(sys::BlumeCapelGraph)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return float(-sys.J * (pair_full ÷ 2)) - _field_sum(sys.h, sys.spins) +
           float(sys.Δ) * sum(s -> Int(s)^2, sys.spins)
end

function _recompute_cached!(sys::BlumeCapelGraph)
    pair_full = 0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full ÷ 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_spin2 = sum(s -> Int(s)^2, sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapelMatrix: sparse J_{ij}, arbitrary topology
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct BlumeCapelMatrix{TJ<:Real, TC<:Real, TH} <: AbstractBlumeCapel
    spins::Vector{Int8}
    const J::SparseMatrixCSC{TJ,Int}
    const Δ::TC
    const h::TH
    cached_pair::Float64    # J-weighted half sum
    cached_mag::Int
    cached_spin2::Int
    cached_field::Float64
end

function BlumeCapelMatrix(J::SparseMatrixCSC{TJ,Int}, Δ::Real; h=0) where {TJ<:Real}
    _check_symmetric(J)
    n = size(J, 1)
    if h isa AbstractVector
        @assert length(h) == n "Field vector length must match number of spins"
        _bc_matrix(J, Δ, collect(float.(h)))
    elseif h isa Real && !iszero(h)
        _bc_matrix(J, Δ, float(h))
    else
        _bc_matrix(J, Δ, NoField())
    end
end

function _bc_matrix(J, Δ, h)
    n = size(J, 1)
    sys = BlumeCapelMatrix{eltype(J), typeof(Δ), typeof(h)}(
        ones(Int8, n), J, Δ, h, 0.0, 0, 0, 0.0)
    _recompute_cached!(sys)
    return sys
end

@inline function neighbor_sum(sys::BlumeCapelMatrix, i)
    acc = 0.0
    @inbounds for ptr in sys.J.colptr[i]:(sys.J.colptr[i+1]-1)
        j = sys.J.rowval[ptr]
        j != i && (acc += sys.J.nzval[ptr] * sys.spins[j])
    end
    return acc
end

@inline function local_pair_interactions(sys::BlumeCapelMatrix, i)
    return sys.spins[i] * neighbor_sum(sys, i)
end

@inline function delta_energy(sys::BlumeCapelMatrix{TJ, TC, TH}, i, dsys::BlumeCapelDelta{Float64}) where {TJ, TC, TH<:Union{Real,AbstractVector}}
    local_term = dsys.coupling + _site_field(sys.h, i)
    return -local_term * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function delta_energy(sys::BlumeCapelMatrix{TJ, TC, NoField}, i, dsys::BlumeCapelDelta{Float64}) where {TJ, TC}
    return -dsys.coupling * dsys.delta_spin + sys.Δ * dsys.delta_spin2
end

@inline function MonteCarloX.modify!(sys::BlumeCapelMatrix{TJ, TC, TH}, i::Int, dsys::BlumeCapelDelta{Float64}) where {TJ, TC, TH}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    _update_field_cache!(sys, sys.h, i, Float64(dsys.delta_spin))
    return nothing
end

@inline function MonteCarloX.modify!(sys::BlumeCapelMatrix{TJ, TC, NoField}, i::Int, dsys::BlumeCapelDelta{Float64}) where {TJ, TC}
    sys.spins[i] = dsys.s_new
    sys.cached_pair += dsys.delta_spin * dsys.coupling
    sys.cached_mag += dsys.delta_spin
    sys.cached_spin2 += dsys.delta_spin2
    return nothing
end

@inline _cached_energy(sys::BlumeCapelMatrix) =
    -sys.cached_pair - _cached_field_energy(sys.h, sys.cached_mag, sys.cached_field) +
    float(sys.Δ) * sys.cached_spin2

function _full_energy(sys::BlumeCapelMatrix)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    return -(pair_full / 2) - _field_sum(sys.h, sys.spins) +
           float(sys.Δ) * sum(s -> Int(s)^2, sys.spins)
end

function _recompute_cached!(sys::BlumeCapelMatrix)
    pair_full = 0.0
    @inbounds for i in eachindex(sys.spins)
        pair_full += local_pair_interactions(sys, i)
    end
    sys.cached_pair = pair_full / 2
    sys.cached_mag = sum(sys.spins)
    sys.cached_spin2 = sum(s -> Int(s)^2, sys.spins)
    sys.cached_field = _field_sum(sys.h, sys.spins)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# BlumeCapel factory constructors
# ═══════════════════════════════════════════════════════════════════════════════

function BlumeCapel(dims::AbstractVector{<:Integer}; J=1, D=0, periodic=true, h=0)
    if J isa Real && periodic
        return BlumeCapelLattice(dims, J, D; h=h)
    end
    graph = Graphs.SimpleGraphs.grid(collect(Int, dims); periodic)
    if J isa Real
        return BlumeCapelGraph(graph, J, D; h=h)
    elseif J isa AbstractVector{<:Real}
        return BlumeCapel(graph, J, D; h=h)
    else
        error("J must be Real or AbstractVector{<:Real}")
    end
end

BlumeCapel(graph::SimpleGraph, J::Real, D::Real; h=0) = BlumeCapelGraph(graph, J, D; h=h)
BlumeCapel(J::SparseMatrixCSC, D::Real; h=0) = BlumeCapelMatrix(J, D; h=h)

function BlumeCapel(graph::SimpleGraph, J::AbstractVector{<:Real}, D::Real; h=0)
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
    n = nv(graph)
    return BlumeCapelMatrix(sparse(rows, cols, vals, n, n), D; h=h)
end
