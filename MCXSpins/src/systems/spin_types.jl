# ── Spin types ────────────────────────────────────────────────────────────────
#
# The "model" axis of a spin system reduces to a SPIN TYPE: the local value set, its storage
# type, and the single-spin proposal. Everything else (couplings, adjacency) lives in the
# interactions (see interactions.jl).

"""
    SpinType

Kind of spin (local degree of freedom) of a spin system: fixes the value set, its storage
type, and the single-spin proposal. Required interface:

    one_state(spintype)                    — reference "all up" state (deterministic init)
    random_state(rng, spintype)            — uniform draw from the local states
    propose_state(rng, spintype, i, s_old) — Monte Carlo proposal for site `i` given its
                                             current state (the site index enables
                                             site-dependent proposals, e.g. dilution or
                                             frozen sites; the spin types here ignore it)

Discrete spin types additionally provide `states(spintype)`, a compile-time tuple of all
local states.
"""
abstract type SpinType end

"""
    Spin(S)

Discrete spin-`S` type in the integer σ-convention (σ = m·denominator(S), m = −S…S), so
`Spin(1//2)` ↦ {−1,+1} (Ising), `Spin(1)` ↦ {−1,0,+1} (Blume–Capel), `Spin(3//2)` ↦ {−3,−1,+1,+3}.
The quantum number is a type parameter: state lists are compile-time tuples, no globals needed.
"""
struct Spin{S} <: SpinType end
# The m = −S…S ladder in unit steps only closes for (half-)integer S: 2S must be a positive
# integer (the state count is 2S+1). Anything else has no σ-convention state set — reject at
# construction with a clear message instead of an InexactError deep in `states`.
function Spin(s::Real)
    isinteger(2s) && s > 0 || throw(ArgumentError(
        "Spin(S) requires a positive integer or half-integer S (got S = $s): " *
        "the ladder m = -S…S in unit steps has 2S+1 states only for integer 2S"))
    return Spin{Rational{Int}(s)}()
end

"""
    states(spintype)

All local states of a discrete spin type, as a compile-time tuple in ascending σ order.
"""
# σ_k = m_k·denominator(S) with m_k = k−1−S: valid for integer and half-integer S alike
# (Spin(1//2) → (−1,+1), Spin(1) → (−1,0,+1), Spin(3//2) → (−3,−1,+1,+3)). A pure function
# of the type parameter, so the ntuple is constant-folded at compile time — calling this in
# a hot loop costs nothing and allocates nothing (the isbits NTuple lives in registers).
@inline states(::Spin{S}) where S =
    ntuple(k -> Int8((2 * (k - 1) - Int(2S)) * denominator(S) ÷ 2), Val(Int(2S + 1)))

one_state(spintype::Spin) = states(spintype)[end]
# Uniform over ALL states — used for :random initialization. Not the Monte Carlo proposal:
# propose_state draws uniformly from the states EXCLUDING the current one.
function random_state(rng::AbstractRNG, spintype::Spin)
    sts = states(spintype)
    return sts[rand(rng, 1:length(sts))]
end

# Generic proposal for ANY Spin{S}: uniform among the 2S other states with a single rand
# (skip-current trick: draw from the first n−1 slots, map a collision with s_old to slot n).
# The state count 2S+1 is a compile-time constant, so this is as good as hand-written cases.
@inline function propose_state(rng::AbstractRNG, spintype::Spin, i, s_old::Int8)
    sts = states(spintype)
    n = length(sts)
    @inbounds s = sts[rand(rng, 1:n-1)]
    return s == s_old ? (@inbounds sts[n]) : s
end

# Spin-1/2: flipping is deterministic. The generic path would still burn one rand() per
# proposal (drawing from 1:1 before the collision branch); skipping the draw is measurable
# in the tight Ising loop and keeps the rng stream of classic single-flip Ising codes.
@inline propose_state(::AbstractRNG, ::Spin{1//2}, i, s_old::Int8) = Int8(-s_old)

"""
    XYSpin()

Continuous planar-rotor spins: unit phasors `ComplexF64`. The rotation proposal takes its
half-width from the update call as a keyword — `spin_flip!(sys, alg; Δθ)` — so step size stays
with the update (adaptable per call), not with the system.
"""
struct XYSpin <: SpinType end
one_state(::XYSpin) = 1.0 + 0.0im
random_state(rng::AbstractRNG, ::XYSpin) = cis(2π * rand(rng))

# Symmetric rotation proposal of half-width Δθ; cis(angle(…)) keeps the modulus at exactly 1.
@inline propose_state(rng::AbstractRNG, ::XYSpin, i, s_old::ComplexF64; Δθ::Real) =
    cis(angle(s_old) + Δθ * (2 * rand(rng) - 1))

"""
    HeisenbergSpin()

Classical Heisenberg spins: unit 3-vectors (`SVector{3,Float64}`), proposed uniformly on
the sphere.
"""
struct HeisenbergSpin <: SpinType end
one_state(::HeisenbergSpin) = SVector(0.0, 0.0, 1.0)
@inline function random_state(rng::AbstractRNG, ::HeisenbergSpin)
    s, c = sincos(2π * rand(rng))
    z = 2.0 * rand(rng) - 1.0
    r = sqrt(1.0 - z * z)
    return SVector(r * c, r * s, z)
end
@inline propose_state(rng::AbstractRNG, spintype::HeisenbergSpin, i, s_old::SVector{3,Float64}) =
    random_state(rng, spintype)

# ── Spin-space inner product: the one hook that makes interactions model-generic ─────────────

@inline _dot(a::Real, b::Real) = a * b
@inline _dot(a::Complex, b::Complex) = real(conj(a) * b)    # = cos(θa − θb) for unit phasors
@inline _dot(a::SVector{3,Float64}, b::SVector{3,Float64}) = dot(a, b)

# ── Storage-type helpers shared by interactions and the system ───────────────────────────────

# Accumulator zero: Int8 spins accumulate in Int — spins stay Int8 for memory locality
# (8× smaller, cache-friendly) while the accumulator is native-width, overflow-safe, and exact.
# Everything else (ComplexF64, SVector, …) accumulates in its own type.
@inline _acc_zero(::Type{Int8}) = 0
@inline _acc_zero(::Type{T}) where T = zero(T)

# The spin sum Σσ and its change under a single-spin move, in the accumulator type
# (Int8 → Int). Live here because they dispatch on the storage types defined above.
_spin_sum(spins::Vector{T}) where T = reduce(+, spins; init=_acc_zero(T))

@inline _spin_sum_delta(s_new::Int8, s_old::Int8) = Int(s_new) - Int(s_old)
@inline _spin_sum_delta(s_new, s_old) = s_new - s_old
