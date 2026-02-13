# Ising Model
# Based on examples/api.ipynb

using Graphs

"""
    Ising{T} <: AbstractSpinSystem

2D Ising model on a graph.

# Fields
- `spins::Vector{Int8}`: Spin configuration (±1)
- `graph::SimpleGraph`: Underlying graph structure
- `nbrs::Vector{Vector{Int}}`: Precomputed neighbor lists
- `J::T`: Coupling constant (can be scalar or vector for inhomogeneous systems)
- `sum_pairs::Int`: Cached sum of sᵢsⱼ over edges
- `sum_spins::Int`: Cached sum of spins

# Hamiltonian
H = -J ∑_{<i,j>} sᵢsⱼ
"""
mutable struct Ising{T} <: AbstractSpinSystem
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    J::T
    sum_pairs::Int
    sum_spins::Int

    function Ising(graph::SimpleGraph, J)
        T = typeof(J)
        if typeof(J) <: Vector{<:Real}
            @assert length(J) == ne(graph) "J vector must match number of edges"
        end

        nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:nv(graph)]

        # Initialize with all spins = +1
        spins = ones(Int8, nv(graph))
        sum_pairs = ne(graph)
        sum_spins = nv(graph)

        new{T}(spins, graph, nbrs, J, sum_pairs, sum_spins)
    end
end

"""
    Ising(dims::Vector{Int}; J=1, periodic=true)

Convenience constructor for Ising model on a hypercubic lattice.

# Arguments
- `dims`: Dimensions of the lattice [Lx, Ly, ...]
- `J`: Coupling constant (default: 1)
- `periodic`: Use periodic boundary conditions (default: true)
"""
function Ising(dims::Vector{Int}; J=1, periodic=true)
    graph = Graphs.SimpleGraphs.grid(dims; periodic)
    return Ising(graph, J)
end

"""
    init!(sys::Ising, type::Symbol; rng=nothing, states=[-1, 1])

Initialize the Ising system.

# Arguments
- `sys`: Ising system to initialize
- `type`: Initialization type (:up, :down, :random)
- `rng`: Random number generator (required for :random)
- `states`: Possible spin states (default: [-1, 1])
"""
function init!(sys::Ising, type::Symbol; rng=nothing, states=[-1, 1])
    if type == :up
        sys.spins .= 1
        sys.sum_pairs = ne(sys.graph)
        sys.sum_spins = length(sys.spins)
    elseif type == :down
        sys.spins .= -1
        sys.sum_pairs = ne(sys.graph)
        sys.sum_spins = -length(sys.spins)
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        for i in eachindex(sys.spins)
            sys.spins[i] = rand(rng, states)
        end
        # Recompute bookkeeping
        sum_pairs = zero(sys.J)
        for i in 1:length(sys.spins)
            sum_pairs += _local_spin_pairs(sys, i)
        end
        sys.sum_pairs = sum_pairs ÷ 2
        sys.sum_spins = sum(sys.spins)
    else
        error("Unknown initialization type: $type")
    end
    return sys
end

# Helper functions

"""
    _local_spin_pairs(sys::Ising, i)

Calculate sum of sᵢsⱼ for site i with all its neighbors.
"""
@inline function _local_spin_pairs(sys::Ising, i)
    s = sys.spins[i]
    acc = 0
    for j in sys.nbrs[i]
        acc += s * sys.spins[j]
    end
    return acc
end

# Observables

"""
    magnetization(sys::Ising)

Calculate absolute magnetization.
"""
@inline magnetization(sys::Ising) = abs(sys.sum_spins)

"""
    energy(sys::Ising)

Calculate total energy.
"""
@inline energy(sys::Ising) = -sys.J * sys.sum_pairs

"""
    delta_energy(sys::Ising, i)

Calculate energy change if spin at site i is flipped.
"""
@inline delta_energy(sys::Ising, i) = 2 * sys.J * _local_spin_pairs(sys, i)

# Updates

"""
    modify!(sys::Ising, i, ΔE)

Flip spin at site i and update cached quantities.

# Arguments
- `sys`: Ising system
- `i`: Site index
- `ΔE`: Energy change from the flip
"""
@inline function modify!(sys::Ising, i, ΔE)
    old = sys.spins[i]
    new = -old
    sys.spins[i] = new

    # Update bookkeeping
    sys.sum_pairs -= ΔE ÷ sys.J
    sys.sum_spins += 2 * new

    return nothing
end

"""
    spin_flip!(sys::Ising, alg::AbstractImportanceSampling)

Perform a single spin flip update using importance sampling.

# Arguments
- `sys`: Ising system
- `alg`: Importance sampling algorithm (e.g., Metropolis)
"""
function spin_flip!(sys::Ising, alg::AbstractImportanceSampling)
    i = rand(alg.rng, 1:length(sys.spins))
    ΔE = delta_energy(sys, i)
    log_ratio = alg.logweight(ΔE)
    if accept!(alg, log_ratio)
        modify!(sys, i, ΔE)
    end
end
