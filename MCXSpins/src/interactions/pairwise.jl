# ── PairInteraction: uniform J on a local neighborhood ────────────────────────
#
# Caches Σ_{<ij>}⟨s_i,s_j⟩ (before J). Deliberately distinct from PairInteractionMatrix:
# folding a uniform J into a sparse matrix would store one redundant float per bond, lose the
# exact Int cache + tabulated acceptance, and lose the fixed-degree NTuple fast path.

"""
    PairInteraction(J, partners)

Pair term −J Σ_{<ij>}⟨s_i,s_j⟩ with one shared coupling `J` on a local neighborhood
structure: `partners[i]` answers "who interacts with site i". The adjacency container is a
type parameter: `Vector{NTuple{NN,Int}}` for fixed-degree lattices (fast path) or
`Vector{Vector{Int}}` for arbitrary graphs — same interaction, two storage layouts.
"""
struct PairInteraction{TJ<:Real, TN, TC} <: AbstractInteraction
    J::TJ
    partners::Vector{TN}
    cache::Cache{TC}
end
PairInteraction(J::Real, partners::Vector) = PairInteraction(J, partners, Cache(0))  # discrete: Int cache

@inline delta(t::PairInteraction, spins, i, s_new) =
    _dot(s_new - (@inbounds spins[i]), partner_sum(t.partners, spins, i))
@inline delta_energy(t::PairInteraction, δ) = -t.J * δ
@inline energy(t::PairInteraction) = -t.J * t.cache.val
# The cache IS filled at construction time — but at the SYSTEM level: an interaction is
# built before any spin vector exists, so SpinSystem(...) calls recompute_all! (which calls
# this) once the spins are allocated. Also the reference path after cluster moves.
function recompute!(t::PairInteraction, spins)
    acc = zero(t.cache.val)
    @inbounds for i in eachindex(spins)
        acc += _dot(spins[i], partner_sum(t.partners, spins, i))
    end
    t.cache.val = _half(acc)
    return nothing
end

partners(t::PairInteraction, i) = t.partners[i]

# Two-site change (Kawasaki): both single-site deltas plus the cross-term correcting the
# bond(s) between i and j, which the two partner sums see from both ends but without the
# joint change.
@inline function delta(t::PairInteraction, spins, ij::NTuple{2,Int}, s_new::NTuple{2})
    i, j = ij
    @inbounds dsi = s_new[1] - spins[i]
    @inbounds dsj = s_new[2] - spins[j]
    δ = _dot(dsi, partner_sum(t.partners, spins, i)) +
        _dot(dsj, partner_sum(t.partners, spins, j))
    nbonds = 0
    @inbounds for k in t.partners[i]
        nbonds += Int(k == j)
    end
    return δ + nbonds * _dot(dsi, dsj)
end
