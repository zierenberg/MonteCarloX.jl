# Blume-Capel Model
# Based on examples/api.ipynb

using Graphs

"""
    BlumeCapel <: AbstractSpinSystem

Blume-Capel model on a graph (spin-1 system with crystal field).

# Fields
- `spins::Vector{Int8}`: Spin configuration (-1, 0, +1)
- `graph::SimpleGraph`: Underlying graph structure
- `nbrs::Vector{Vector{Int}}`: Precomputed neighbor lists
- `states::Vector{Int8}`: Possible spin states [-1, 0, 1]
- `J::Real`: Exchange coupling constant
- `D::Real`: Crystal field parameter
- `sum_pairs::Int`: Cached sum of sᵢsⱼ over edges
- `sum_spins::Int`: Cached sum of spins
- `sum_spins2::Int`: Cached sum of sᵢ²

# Hamiltonian
H = -J ∑_{<i,j>} sᵢsⱼ + D ∑ᵢ sᵢ²
"""
mutable struct BlumeCapel <: AbstractSpinSystem
    spins::Vector{Int8}
    graph::SimpleGraph
    nbrs::Vector{Vector{Int}}
    states::Vector{Int8}
    J::Real
    D::Real
    sum_pairs::Int
    sum_spins::Int
    sum_spins2::Int

    function BlumeCapel(graph::SimpleGraph, J, D)
        nbrs = [collect(Graphs.neighbors(graph, i)) for i in 1:nv(graph)]
        states = Int8[-1, 0, 1]

        # Initialize with all spins = +1
        spins = ones(Int8, nv(graph))
        sum_pairs = ne(graph)
        sum_spins = nv(graph)
        sum_spins2 = nv(graph)

        new(spins, graph, nbrs, states, J, D, sum_pairs, sum_spins, sum_spins2)
    end
end

"""
    BlumeCapel(dims::Vector{Int}; J=1, D=0, periodic=true)

Convenience constructor for Blume-Capel model on a hypercubic lattice.

# Arguments
- `dims`: Dimensions of the lattice [Lx, Ly, ...]
- `J`: Exchange coupling (default: 1)
- `D`: Crystal field parameter (default: 0)
- `periodic`: Use periodic boundary conditions (default: true)
"""
function BlumeCapel(dims::Vector{Int}; J=1, D=0, periodic=true)
    graph = Graphs.SimpleGraphs.grid(dims; periodic)
    return BlumeCapel(graph, J, D)
end

"""
    init!(sys::BlumeCapel, type::Symbol; rng=nothing)

Initialize the Blume-Capel system.

# Arguments
- `sys`: BlumeCapel system to initialize
- `type`: Initialization type (:up, :down, :zero, :random)
- `rng`: Random number generator (required for :random)
"""
function init!(sys::BlumeCapel, type::Symbol; rng=nothing)
    if type == :up
        sys.spins .= 1
        sys.sum_pairs = ne(sys.graph)
        sys.sum_spins = length(sys.spins)
        sys.sum_spins2 = length(sys.spins)
    elseif type == :down
        sys.spins .= -1
        sys.sum_pairs = ne(sys.graph)
        sys.sum_spins = -length(sys.spins)
        sys.sum_spins2 = length(sys.spins)
    elseif type == :zero
        sys.spins .= 0
        sys.sum_pairs = 0
        sys.sum_spins = 0
        sys.sum_spins2 = 0
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        for i in eachindex(sys.spins)
            sys.spins[i] = rand(rng, sys.states)
        end
        # Recompute bookkeeping
        sum_pairs = 0
        for i in 1:length(sys.spins)
            sum_pairs += local_pair_interactions(sys, i)
        end
        sys.sum_pairs = sum_pairs ÷ 2
        sys.sum_spins = sum(sys.spins)
        sys.sum_spins2 = sum(s -> s^2, sys.spins)
    else
        error("Unknown initialization type: $type")
    end
    return sys
end

# Observables

"""
    magnetization(sys::BlumeCapel)

Calculate absolute magnetization.
"""
@inline magnetization(sys::BlumeCapel) = abs(sys.sum_spins)

"""
    energy(sys::BlumeCapel)

Calculate total energy.
"""
@inline energy(sys::BlumeCapel) = -sys.J * sys.sum_pairs + sys.D * sys.sum_spins2

"""
    delta_energy(sys::BlumeCapel, i, s_new)

Calculate energy change if spin at site i changes to s_new.
"""
@inline function delta_energy(sys::BlumeCapel, i, s_new)
    s_old = sys.spins[i]
    ΔE_exchange = sys.J * (s_old - s_new) * local_pair_interactions(sys, i)
    ΔE_field = sys.D * (s_new^2 - s_old^2)
    return ΔE_exchange + ΔE_field
end

# Updates

"""
    modify!(sys::BlumeCapel, i, s_new, ΔE)

Change spin at site i to s_new and update cached quantities.

# Arguments
- `sys`: BlumeCapel system
- `i`: Site index
- `s_new`: New spin value
- `ΔE`: Energy change from the update
"""
@inline function modify!(sys::BlumeCapel, i, s_new, ΔE)
    s_old = sys.spins[i]
    sys.spins[i] = s_new

    # Update bookkeeping
    Δ_pairs = (s_new - s_old) * local_pair_interactions(sys, i) ÷ 2
    sys.sum_pairs += Δ_pairs
    sys.sum_spins += (s_new - s_old)
    sys.sum_spins2 += (s_new^2 - s_old^2)

    return nothing
end

"""
    spin_flip!(sys::BlumeCapel, alg::AbstractImportanceSampling)

Perform a single spin update using importance sampling.

# Arguments
- `sys`: BlumeCapel system
- `alg`: Importance sampling algorithm (e.g., Metropolis)
"""
function spin_flip!(sys::BlumeCapel, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = rand(alg.rng, sys.states)
    
    if s_new != sys.spins[i]
        ΔE = delta_energy(sys, i, s_new)
        E_old = energy(sys)
        E_new = E_old + ΔE
        log_ratio = log_acceptance_ratio(alg, E_new, E_old)
        if accept!(alg, log_ratio)
            modify!(sys, i, s_new, ΔE)
        end
    end
end

function spin_flip!(sys::BlumeCapel, alg::AbstractMetropolis)
    i = pick_site(alg.rng, length(sys.spins))
    s_new = rand(alg.rng, sys.states)

    if s_new != sys.spins[i]
        ΔE = delta_energy(sys, i, s_new)
        log_ratio = log_acceptance_ratio(alg, ΔE)
        if accept!(alg, log_ratio)
            modify!(sys, i, s_new, ΔE)
        end
    end
end
