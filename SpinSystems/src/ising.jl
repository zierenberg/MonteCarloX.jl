"""
    AbstractIsing <: AbstractSpinSystem

Base type for Ising-like models with simplified spin flip interface.

Implementations must provide:
- `delta_energy(sys, i)`: Energy change from flipping spin \$i\$
- `modify!(sys, i, ΔE)`: Apply spin flip and update cached quantities
"""
abstract type AbstractIsing <: AbstractSpinSystem end

using Graphs
"""
    Ising{T} <: AbstractIsing

Ising model on a graph.

# Fields
- `spins::Vector{Int8}`: Spin configuration \$(\\pm 1)\$
- `graph::SimpleGraph`: Underlying graph structure
- `nbrs::Vector{Vector{Int}}`: Precomputed neighbor lists (this is done during construction)
- `J::T`: Coupling constant \$J\$ (can be scalar or vector for inhomogeneous systems)
- `sum_pairs::Int`: Cached sum of \$\\sum_{\\langle i,j \\rangle} s_i s_j\$ over edges
- `sum_spins::Int`: Cached sum of \$\\sum_i s_i\$

# Hamiltonian
\$ H = -J \\sum_{\\langle i,j \\rangle} s_i s_j - h \\sum_i s_i \$
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
- `dims`: Dimensions of the lattice \$[L_x, L_y, \\ldots]\$
- `J`: Coupling constant \$J\$ (default: 1)
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
- `type`: Initialization type (`:up`, `:down`, `:random`)
- `rng`: Random number generator (required for `:random`)
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

Calculate absolute magnetization \$|M| = |\\sum_i s_i|\$.
"""
@inline magnetization(sys::Ising) = abs(sys.sum_spins)

"""
    energy(sys::Ising; full=false)

Calculate total energy \$E = -J \\sum_{\\langle i,j \\rangle} s_i s_j\$.

If `full=false` (default), returns the cached energy. If `full=true`, recomputes from scratch by summing over all spins.
"""
function energy(sys::Ising; full=false)
    if full
        sum_pairs = zero(sys.J)
        for i in 1:length(sys.spins)
            sum_pairs += local_spin_pairs(sys, i)
        end
        return -sys.J * (sum_pairs ÷ 2)
    else
        return -sys.J * sys.sum_pairs
    end
end

"""
    delta_energy(sys::Ising, i)

Calculate energy change \$\\Delta E\$ if spin \$s_i\$ at site \$i\$ is flipped.
"""
@inline delta_energy(sys::Ising, i) = 2 * sys.J * local_spin_pairs(sys, i)

"""
    modify!(sys::Ising, i, ΔE)

Flip spin \$s_i\$ at site \$i\$ and update cached quantities.

# Arguments
- `sys`: Ising system
- `i`: Site index
- `ΔE`: Energy change \$\\Delta E\$ from the flip
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

Perform a single spin flip update for Ising-like models using importance sampling.

# Arguments
- `sys::AbstractIsing`: Ising model to update
- `alg::AbstractImportanceSampling`: Algorithm with RNG and log weight function \$\\log w(\\Delta E)\$
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

Fast 2D Ising model with \$J=1\$, periodic boundary conditions and on-the-fly neighbor calculation.

# Fields
- `spins::Vector{Int8}`: Spins \$(\\pm 1)\$ in row-major order
- `Lx::Int`: Lattice width \$L_x\$
- `Ly::Int`: Lattice height \$L_y\$
- `energy::Int`: Cached total energy \$E\$
- `magnetization::Int`: Cached total magnetization \$M\$
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
    full_energy(sys::Ising_2Dgrid_optim)

Compute the full total energy \$E = -\\sum_{\\langle i,j \\rangle} s_i s_j\$ by iterating over all sites and summing local contributions.
"""
function full_energy(sys::Ising_2Dgrid_optim)
    energy_total = 0
    for i in 1:length(sys.spins)
        energy_total += local_energy(sys, i)
    end
    return -energy_total ÷ 2  # Divide by 2 because each bond is counted twice
end

"""
    local_energy(sys::Ising_2Dgrid_optim, i)

Calculate the local energy contribution \$E_i = \\sum_{j \\in \\text{nbrs}(i)} s_i s_j\$ for site \$i\$ with its neighbors (on-the-fly calculation).
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

"""
    energy(sys::Ising_2Dgrid_optim; full=false)

Get the energy of the 2D Ising system.

If `full=false` (default), returns the cached energy. If `full=true`, recomputes from scratch.
"""
function energy(sys::Ising_2Dgrid_optim; full=false)
    if full
        return full_energy(sys)
    else
        return sys.energy
    end
end

"""
    magnetization(sys::Ising_2Dgrid_optim; full=false)

Get the magnetization of the 2D Ising system.

If `full=false` (default), returns the cached magnetization. If `full=true`, recomputes from scratch.
"""
function magnetization(sys::Ising_2Dgrid_optim; full=false)
    if full
        return abs(sum(sys.spins))
    else
        return abs(sys.magnetization)
    end
end

@inline delta_energy(sys::Ising_2Dgrid_optim, i) = 2 * local_energy(sys, i)

function modify!(sys::Ising_2Dgrid_optim, i, ΔE)
    sys.spins[i] = -sys.spins[i]
    sys.energy += ΔE
    sys.magnetization += 2 * sys.spins[i]
    return nothing
end
