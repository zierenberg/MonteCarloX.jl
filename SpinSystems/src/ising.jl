"""
    AbstractIsing <: AbstractSpinSystem

Base type for Ising-like models with simplified spin flip interface.

Implementations must provide:
- `delta_energy(sys, i)`: Energy change from flipping spin i
- `modify!(sys, i, ΔE)`: Apply spin flip and update cached quantities
"""
abstract type AbstractIsing <: AbstractSpinSystem end

using Graphs
"""
    Ising{T} <: AbstractIsing

Ising model on a graph.

# Fields
- `spins::Vector{Int8}`: Spin configuration (±1)
- `graph::SimpleGraph`: Underlying graph structure
- `nbrs::Vector{Vector{Int}}`: Precomputed neighbor lists
- `J::T`: Coupling constant (can be scalar or vector for inhomogeneous systems)
- `sum_pairs::Int`: Cached sum of sᵢsⱼ over edges
- `sum_spins::Int`: Cached sum of spins

# Hamiltonian
\$ H = -J \\sum_{⟨i,j⟩} s_i s_j - h \\sum_i s_i \$
"""
mutable struct Ising{T} <: AbstractIsing
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
    init!(sys::Ising, type::Symbol; rng=nothing)

Initialize the Ising system.

# Arguments
- `sys`: Ising system to initialize
- `type`: Initialization type (:up, :down, :random)
- `rng`: Random number generator (required for :random)
"""
function init!(sys::Ising, type::Symbol; rng=nothing)
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
            sys.spins[i] = rand(rng, [-1,1])
        end
        # Recompute bookkeeping
        sum_pairs = zero(sys.J)
        for i in 1:length(sys.spins)
            sum_pairs += local_spin_pairs(sys, i)
        end
        sys.sum_pairs = sum_pairs ÷ 2
        sys.sum_spins = sum(sys.spins)
    else
        error("Unknown initialization type: $type")
    end
    return sys
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
@inline delta_energy(sys::Ising, i) = 2 * sys.J * local_spin_pairs(sys, i)

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
    spin_flip!(sys::AbstractIsing, alg::AbstractImportanceSampling)

Perform a single spin flip update for Ising-like models.

# Arguments
- `sys::AbstractIsing`: Ising model to update
- `alg::AbstractImportanceSampling`: Algorithm with RNG and log weight function
"""
function spin_flip!(sys::AbstractIsing, alg::AbstractImportanceSampling)
    i = pick_site(alg.rng, length(sys.spins))
    ΔE = delta_energy(sys, i)
    log_ratio = alg.logweight(ΔE)
    if accept!(alg, log_ratio)
        modify!(sys, i, ΔE)
    end
    return nothing
end


# ============================================================================
# Optimized 2D Ising Model
# ============================================================================

"""
    Ising_2Dgrid_optim <: AbstractIsing

Fast 2D Ising model with J=1, periodic boundary conditions and on-the-fly neighbor calculation.

# Fields
- `spins::Vector{Int8}`: Spins (±1) in row-major order
- `Lx::Int`: Lattice width
- `Ly::Int`: Lattice height  
- `energy::Int`: Cached total energy
- `magnetization::Int`: Cached total magnetization
"""
mutable struct Ising_2Dgrid_optim <: AbstractIsing
    spins::Vector{Int8}
    const Lx::Int8
    const Ly::Int8
    energy::Int
    magnetization::Int
    
    function Ising_2Dgrid_optim(Lx::Int, Ly::Int)
        N = Lx * Ly
        # All spins up: each site has 4 neighbors, so -sum_pairs = -2N
        new(ones(Int8, N), Lx, Ly, -2N, N)
    end
end

"""
    local_energy(sys::Ising_2Dgrid_optim, i)

Calculate the sum of sᵢsⱼ for site i with its neighbors (on-the-fly calculation).
"""
@inline function local_energy(sys::Ising_2Dgrid_optim, i)
    s = sys.spins[i]
    acc = 0
    
    x = ((i - 1) % sys.Lx) + 1
    y = div(i - 1, sys.Lx) + 1
    
    x_left = x == 1 ? sys.Lx : x - 1
    x_right = x == sys.Lx ? 1 : x + 1
    y_up = y == 1 ? sys.Ly : y - 1
    y_down = y == sys.Ly ? 1 : y + 1
    
    @inbounds begin
        acc += s * sys.spins[x_left + (y - 1) * sys.Lx]
        acc += s * sys.spins[x_right + (y - 1) * sys.Lx]
        acc += s * sys.spins[x + (y_up - 1) * sys.Lx]
        acc += s * sys.spins[x + (y_down - 1) * sys.Lx]
    end

    return acc
end

@inline delta_energy(sys::Ising_2Dgrid_optim, i) = 2 * local_energy(sys, i)

function modify!(sys::Ising_2Dgrid_optim, i, ΔE)
    sys.spins[i] = -sys.spins[i]
    sys.energy += ΔE
    sys.magnetization += 2 * sys.spins[i]
    return nothing
end
