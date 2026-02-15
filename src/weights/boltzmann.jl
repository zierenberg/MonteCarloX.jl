# Boltzmann Log Weight
# For canonical ensemble sampling at fixed temperature

"""
    BoltzmannLogWeight <: AbstractLogWeight

Boltzmann weight function: w(E) = exp(-β*E).

This is the standard weight for canonical ensemble Monte Carlo simulations
at a fixed inverse temperature β = 1/(k_B T).

# Fields
- `β::Real`: Inverse temperature

# Examples
```julia
# Create weight for temperature T=2.0 (assuming k_B = 1)
logweight = BoltzmannLogWeight(0.5)

# Evaluate log weight for energy E
logw = logweight(energy)  # Returns -β * energy
```
"""
struct BoltzmannLogWeight <: AbstractLogWeight
    β::Real
end

"""
    (lw::BoltzmannLogWeight)(E)

Evaluate log weight: log(w) = -β * sum(E).

E can be a scalar energy or a vector/tuple of energy components.
If E is a collection, the sum of all components is used.

# Examples
```julia
logweight = BoltzmannLogWeight(1.0)
logweight(-2.5)              # Returns 2.5
logweight([-1, -2, -3])      # Returns 6.0
logweight((H=-10, M=2))      # Returns 8.0 (from named tuple)
```
"""
@inline (lw::BoltzmannLogWeight)(E) = -lw.β * sum(E)
