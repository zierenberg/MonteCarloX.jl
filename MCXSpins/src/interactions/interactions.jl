# ── Interactions: the energy terms of a spin system ───────────────────────────
#
# An INTERACTION is one energy term; its COUPLINGS are the constants inside (J, h, Δ, κ);
# the PARTNERS of site i are the sites the interaction connects it to. Each interaction owns
# (a) its couplings, (b) its own index structure (partner tables, sparse J_ij — there is no
# system-level "topology" axis), and (c) a CACHE holding the term's value BEFORE the coupling
# is applied (Σ_{<ij>}σσ before J, Σσ before h, Σσ² before Δ). For discrete spins these sums
# are exact integers: caches never drift, total energy is O(1), and with integer J the whole
# ΔE chain stays in Int — enabling tabulated acceptance with zero exp() calls (see
# MCXSpins/benchmarks/benchmark_mcxspins.jl).
#
# CHANGE PROTOCOL: a proposed spin change implies, for every interaction term, a DELTA — the
# term's own change record (for cached terms the change of the coupling-free sum, for
# cache-free asymmetric terms the dynamical ΔE itself). Deltas are computed exactly ONCE per
# proposal and flow through the generic spin_flip! hooks (delta_sys → delta_energy →
# accept! → modify!), so no partner sum is ever evaluated twice. Every interaction
# implements
#
#     delta(t, spins, i, s_new)  — the term's delta for setting spin i to s_new
#     delta_energy(t, δ)         — ΔE implied by δ (couplings are applied HERE)
#     commit!(t, δ)              — apply δ on acceptance (default: cache.val += δ;
#                                  no-op override for cache-free terms)
#     energy(t)                  — O(1) from the cache
#     recompute!(t, spins)       — O(N) cache rebuild (reference)
#
# plus, optionally, the two-site variant delta(t, spins, (i,j), (si',sj')) for Kawasaki
# exchange. Each function extends to the whole interaction tuple by multiple dispatch
# (compile-time unrolled recursion — no parallel underscore helpers).
#
# SYMMETRY TRAIT: every interaction declares symmetry() ∈ {SymmetricCoupling,
# AsymmetricCoupling}. Symmetric terms derive from a proper Hamiltonian; their caches are
# exact energy bookkeeping and they contribute to hamiltonian_energy(sys). Asymmetric terms
# break detailed balance and have NO Hamiltonian — they are CACHE-FREE, and energy(t) throws.
# Energy-like quantities for such systems (e.g. ½ΣᵢEᵢ) are OBSERVABLES: compute them from the
# spins in measurement code, they do not belong in the system.

abstract type AbstractInteraction end

struct SymmetricCoupling end     # proper Hamiltonian term: cache = exact energy bookkeeping
struct AsymmetricCoupling end    # detailed balance broken: no Hamiltonian, no cache

symmetry(::AbstractInteraction) = SymmetricCoupling()

# Cache holding the term's coupling-free sum (Int-exact for discrete spins).
mutable struct Cache{T}
    val::T
end

# Default commit: add the term's delta to its cache. Cache-free interactions override
# with a no-op.
@inline commit!(t::AbstractInteraction, δ) = (t.cache.val += δ; nothing)

# Neighborhood sum over any adjacency container (NTuple or Vector entries), one method for all
# spin types.
@inline function partner_sum(partners::Vector, spins::Vector{T}, i) where T
    nb = @inbounds partners[i]
    acc = _acc_zero(T)
    @inbounds for j in nb
        acc += spins[j]
    end
    return acc
end

# Halve the double-counted pair sum: exact (÷) for Int caches, float division otherwise.
@inline _half(x::Integer) = x ÷ 2
@inline _half(x) = x / 2

# ── Tuple extension by multiple dispatch ──────────────────────────────────────
#
# The per-term functions extend to interaction tuples of ANY length by structural recursion:
# `first(ints)` handles term 1 and the call on `Base.tail(ints)` (the remaining N−1 terms)
# recurses until a base case. The tuple length is a type parameter, so the compiler unrolls
# the recursion completely — allocation-free, fixed left-to-right order, no runtime loop.
#
# Two kinds of base case appear below, deliberately:
#   • empty tuple `Tuple{}` where the neutral element is type-agnostic ((), nothing);
#   • ONE-element tuple `Tuple{Any}` for the summing functions, so the sum of an all-Int
#     chain stays Int (an empty base case would inject `+ 0.0` and promote — losing the
#     tabulated-acceptance property) .

@inline delta(ints::Tuple, spins, i, s_new) =
    (delta(first(ints), spins, i, s_new), delta(Base.tail(ints), spins, i, s_new)...)
@inline delta(::Tuple{}, spins, i, s_new) = ()

@inline delta_energy(ints::Tuple{Any}, δs::Tuple{Any}) = delta_energy(ints[1], δs[1])
@inline delta_energy(ints::Tuple, δs::Tuple) =
    delta_energy(first(ints), first(δs)) + delta_energy(Base.tail(ints), Base.tail(δs))

@inline commit!(ints::Tuple, δs::Tuple) =
    (commit!(first(ints), first(δs)); commit!(Base.tail(ints), Base.tail(δs)))
@inline commit!(::Tuple{}, ::Tuple{}) = nothing

@inline energy(ints::Tuple{Any}) = energy(ints[1])
@inline energy(ints::Tuple) = energy(first(ints)) + energy(Base.tail(ints))

# Hamiltonian bookkeeping: only symmetric interactions constitute a proper energy. This is
# the quantity multicanonical / reweighting methods may use (e.g. muca in the crystal-field
# term of the nonreciprocal Blume–Capel, where that term IS a proper function of state).
# (Float base case is fine here: the result mixes couplings and is Float anyway.)
@inline hamiltonian_energy(ints::Tuple) =
    hamiltonian_energy(first(ints)) + hamiltonian_energy(Base.tail(ints))
@inline hamiltonian_energy(::Tuple{}) = 0.0
@inline hamiltonian_energy(t::AbstractInteraction) = hamiltonian_energy(symmetry(t), t)
@inline hamiltonian_energy(::SymmetricCoupling, t) = energy(t)
@inline hamiltonian_energy(::AsymmetricCoupling, t) = 0.0

# ── Partner queries (setup/observable utilities — not hot-path) ───────────────

"""
    partners(t, i)
    partners(sys, i)

Sites that interaction `t` (or any interaction of `sys`) connects to site `i`. On-site terms
return `()`. The system-level query concatenates the terms' partner lists WITHOUT
deduplication (two terms may both list a site). For observables, cluster construction, and
parallel scheduling; the dynamics itself never calls this.
"""
partners(::AbstractInteraction, i) = ()                       # on-site terms have no partners

partners(ints::Tuple, i) = (partners(first(ints), i)..., partners(Base.tail(ints), i)...)
partners(::Tuple{}, i) = ()
