# Canonical Ensemble Log Weight (Boltzmann weight)
# For canonical ensemble sampling at fixed temperature

"""
    BoltzmannLogWeight <: AbstractLogWeight

Boltzmann weight function: w(E) = exp(-β*E).

This is the standard weight for canonical ensemble Monte Carlo simulations
at a fixed inverse temperature β = 1/(k_B T).

# Fields
- `β::Float64`: Inverse temperature

# Examples
```julia
# Create weight for temperature T=2.0 (assuming k_B = 1)
logweight = BoltzmannLogWeight(0.5)

# Evaluate log weight for scalar energy
logw = logweight(-2.5)  # Returns -0.5 * (-2.5) = 1.25

# Or for vector of energy terms
logw = logweight([-1, -2, -3])  # Returns -0.5 * (-6) = 3.0
```
"""
struct BoltzmannLogWeight{T<:Real} <: AbstractLogWeight
    β::T
end

"""
    (lw::BoltzmannLogWeight)(E)

Evaluate log weight for energy term(s).

For scalar energy: log(w) = -β * E
For array/tuple of terms: log(w) = -β * sum(E)

Specialized for type stability with scalar inputs.

# Arguments
- `E`: Energy (scalar) or energy components (array/tuple)

# Returns
- Log weight as Float64

# Examples
```julia
logweight = BoltzmannLogWeight(1.0)
logweight(-2.5)              # Returns 2.5 (scalar)
logweight([-1, -2, -3])      # Returns 6.0 (array)
```
"""
@inline (lw::BoltzmannLogWeight)(E::Real) = -lw.β * E # Real type ensures that E can also be integer
@inline (lw::BoltzmannLogWeight)(E::AbstractArray) = -lw.β * sum(E)