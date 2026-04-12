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

@inline logweight(e::BoltzmannEnsemble, E::Real) = -e.beta * E
@inline logweight(e::BoltzmannEnsemble, E::AbstractArray) = -e.beta * sum(E)
@inline logweight(e::BoltzmannEnsemble) = x -> logweight(e, x)