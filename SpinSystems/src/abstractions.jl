"""
    AbstractSpinSystem <: AbstractSystem

Base type for spin systems.

A spin system typically contains:
- Spins/states configuration
- Graph/lattice structure
- Interaction parameters
- Cached quantities for efficient updates
"""
abstract type AbstractSpinSystem <: AbstractSystem end

"""
    pick_site(rng, N)

Randomly pick a site index from 1 to N.
"""
@inline pick_site(rng, N) = rand(rng, UInt) % N + 1

"""
    local_pair_interactions(sys::AbstractSpinSystem, i)

Calculate the sum of sᵢsⱼ products for site i with all its neighbors.

This represents the local contribution to pair interactions.

# Arguments
- `sys::AbstractSpinSystem`: Spin system with `spins` and `nbrs` fields
- `i`: Site index

# Returns
- Sum of pair interactions: ∑ⱼ∈neighbors(i) sᵢsⱼ

# Example
```julia
system = Ising([10, 10])
interactions = local_pair_interactions(system, 5)
```
"""
@inline function local_pair_interactions(sys::AbstractSpinSystem, i)
    s = sys.spins[i]
    acc = 0
    for j in sys.nbrs[i]
        acc += s * sys.spins[j]
    end
    return acc
end

# General spin flip update
# This is the standard update for spin systems that can be customized by specific models

"""
    spin_flip!(sys::AbstractSpinSystem, alg::AbstractImportanceSampling)

Perform a generic single spin flip update using importance sampling.

This is a general update function that works for any spin system that implements:
- `hamiltonian_terms(sys)`: Returns energy components (can be scalar, tuple, or named tuple)
- `delta_hamiltonian_terms(sys, i, s_new)`: Returns change in energy components
- `modify!(sys, i, s_new, ΔH)`: Applies the spin change and updates cached quantities
- `sys.states`: Vector of possible spin states

The energy terms are passed through the log weight function, allowing for:
- Canonical sampling: logweight = -β * sum(H)
- Generalized ensembles: logweight can depend on individual terms differently
- Multicanonical sampling: logweight based on custom weight functions

# Arguments
- `sys::AbstractSpinSystem`: Spin system to update
- `alg::AbstractImportanceSampling`: Algorithm with RNG and log weight function

# Example
```julia
# For a simple Ising model with uniform temperature
system = Ising([10, 10])
alg = Metropolis(rng, β=1.0)
spin_flip!(system, alg)

# For Blume-Capel with crystal field
system = BlumeCapel([10, 10], J=1.0, D=0.5)
alg = Metropolis(rng, β=2.0)
spin_flip!(system, alg)
```

# Notes
- Systems can override this with specialized implementations for better performance
- The energy terms can be vectors/tuples to separate different contributions
  (e.g., exchange, field, crystal field terms)
"""
function spin_flip!(sys::AbstractSpinSystem, alg::AbstractImportanceSampling)
    # Pick a random spin site
    i = pick_site(alg.rng, length(sys.spins))
    
    # System defines the possible spin states
    s_new = rand(alg.rng, sys.states)
    
    # Get current Hamiltonian terms and the change from the proposed update
    H_terms = hamiltonian_terms(sys)
    ΔH_terms = delta_hamiltonian_terms(sys, i, s_new)
    
    # Evaluate log weight before and after
    # Terms can be scalars, tuples, or named tuples - sum handles all cases
    log_ratio = alg.logweight(H_terms .+ ΔH_terms) - alg.logweight(H_terms)
    
    # Accept or reject the move
    if accept!(alg, log_ratio)
        modify!(sys, i, s_new, ΔH_terms)
    end
end