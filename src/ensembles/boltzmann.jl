"""
    BoltzmannEnsemble

Canonical-ensemble score with
`logweight(E) = -beta * E`.
"""
struct BoltzmannEnsemble{T<:Real} <: AbstractEnsemble
    beta::T

    function BoltzmannEnsemble(; beta=nothing, β=nothing, T=nothing)
        if beta !== nothing && β !== nothing
            throw(ArgumentError("Specify only one of `beta` or `β`"))
        end

        b = beta === nothing ? β : beta

        if (b === nothing) == (T === nothing)
            throw(ArgumentError("Specify exactly one of `beta`/`β` or `T`"))
        elseif b !== nothing
            return new{typeof(b)}(b)
        else
            val = inv(T)
            return new{typeof(val)}(val)
        end
    end
end

linear_logweight(::BoltzmannEnsemble) = true

# `iszero(E)` short-circuits before the multiplication: at β = 1/T = Inf (T = 0, a zero-temperature
# "always downhill, never uphill" limit), `-beta * E` hits `Inf * 0.0 = NaN` for exactly the E == 0
# case that should trivially contribute zero regardless of beta.
@inline logweight(e::BoltzmannEnsemble, E::Real) = iszero(E) ? zero(float(E)) : -e.beta * E
@inline function logweight(e::BoltzmannEnsemble, E::AbstractArray)
    s = sum(E)
    return iszero(s) ? zero(float(s)) : -e.beta * s
end
@inline logweight(e::BoltzmannEnsemble) = x -> logweight(e, x)