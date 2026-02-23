# Wang-Landau generalized-ensemble scaffolding

using Random

"""
    WangLandau <: AbstractImportanceSampling

Wang-Landau generalized-ensemble sampler.

Notation used here:
- `g(E)`: density of states estimate
- `S(E) = log g(E)`: entropy estimate
- `logWeight(E) = S(E)` stored in `alg.logweight`
- `logf = log(f)`: logarithmic modification factor

Each visit to an energy bin `E_i` applies the local update
`S(E_i) <- S(E_i) + logf`.
In this API convention this appears as
`alg.logweight[E_i] -= logf`.
"""
mutable struct WangLandau{LW,RNG<:AbstractRNG} <: AbstractGeneralizedEnsemble
    rng::RNG
    logweight::LW
    logf::Float64
    steps::Int
    accepted::Int
end

WangLandau(rng::AbstractRNG, logweight::TabulatedLogWeight; logf::Real = 1.0) =
    WangLandau(rng, logweight, Float64(logf), 0, 0)

WangLandau(logweight::TabulatedLogWeight; logf::Real = 1.0) =
    WangLandau(Random.GLOBAL_RNG, logweight; logf=logf)

# dispatch accept! to add Wang-Landau specific bookkeeping
function accept!(alg::WangLandau, x_new::Real, x_old::Real)
    log_ratio = alg.logweight(x_new) - alg.logweight(x_old)
    accepted = _accept!(alg, log_ratio)
    if accepted
        alg.logweight[x_new] -= alg.logf
    else
        alg.logweight[x_old] -= alg.logf
    end
    return accepted
end


"""
    update_weight!(alg::WangLandau, x)

Apply one local Wang-Landau update at `x`:
- `alg.logweight[x] -= alg.logf`

Mutates in place and returns `nothing`.
"""
function update_weight!(
    alg::WangLandau,
    x::Real,
)
    log_weight = alg.logweight
    w = log_weight[x]
    if w === missing
        throw(DomainError(x, "`x` is outside the tabulated log-weight bins"))
    end
    log_weight[x] = w - alg.logf

    return nothing
end

"""
    update_f!(alg::WangLandau)

Update Wang-Landau schedule in-place with
`logf <- 0.5 * logf` (equivalent to `f <- sqrt(f)`).
"""
function update_f!(alg::WangLandau)
    alg.logf *= 0.5
    return nothing
end
