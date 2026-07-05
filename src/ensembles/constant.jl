"""
    ConstantEnsemble(c=0.0)

Ensemble with a constant log-weight, `logweight(x) = c` for every `x` — i.e. the
absence of weighting. Reweighting *to* a `ConstantEnsemble` strips the source
ensemble's bias and recovers the unbiased density: the density of states from a
multicanonical run, a `β=0` Boltzmann target, or the removal of a single-variable
bias. It is the default target of [`reweight`](@ref).

Equivalent to `BoltzmannEnsemble(β=0)` and to `FunctionEnsemble(_ -> c)`.
"""
struct ConstantEnsemble{T<:Real} <: AbstractEnsemble
    c::T
end

ConstantEnsemble() = ConstantEnsemble(0.0)

# A constant log-weight satisfies the linearity identity
# `logweight(Δ) == logweight(a+Δ) - logweight(a)` only when `c == 0` (since it reduces
# to `c == c - c`), so only the zero constant is usable with Metropolis-family algorithms.
linear_logweight(e::ConstantEnsemble) = iszero(e.c)

@inline logweight(e::ConstantEnsemble, x) = e.c
@inline logweight(e::ConstantEnsemble) = _ -> e.c
