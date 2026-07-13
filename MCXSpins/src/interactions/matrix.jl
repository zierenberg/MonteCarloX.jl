# ── PairInteractionMatrix: arbitrary sparse J_ij ──────────────────────────────
#
# Weighted-pair structure answering "how strongly does each pair couple": spin glasses,
# Hopfield, arbitrary J_ij. Caches ½Σ⟨s_i, w_i⟩ with w_i = Σ_j J_ji s_j. Symmetry is
# auto-detected; directed (asymmetric) couplings are allowed but the cache is DEACTIVATED —
# updating it on a flip of i needs the couplings i exerts on others (Jᵀ column), which the
# CSC layout cannot deliver cheaply. Convention: column i holds the influences ON i.

"""
    PairInteractionMatrix(J::SparseMatrixCSC)

Pair term with an arbitrary sparse coupling matrix: −½ Σ_{ij} J_ij ⟨s_i,s_j⟩ (each bond
counted once). Column `i` holds the couplings acting ON site `i`. Asymmetric `J` is allowed
(directed couplings, no Hamiltonian): auto-detected, `energy` then throws and the dynamics
uses the local rule only.
"""
struct PairInteractionMatrix{TJ<:Real, S} <: AbstractInteraction
    J::SparseMatrixCSC{TJ,Int}
    cache::Cache{Float64}
end
function PairInteractionMatrix(J::SparseMatrixCSC)
    return issymmetric(J) ? PairInteractionMatrix{eltype(J), SymmetricCoupling}(J, Cache(0.0)) :
                            PairInteractionMatrix{eltype(J), AsymmetricCoupling}(J, Cache(NaN))
end

symmetry(::PairInteractionMatrix{TJ,S}) where {TJ,S} = S()

# J-weighted neighbor sum over a CSC column.
@inline function partner_sum(J::SparseMatrixCSC, spins, i)
    acc = zero(eltype(J)) * zero(eltype(spins))
    @inbounds for ptr in J.colptr[i]:(J.colptr[i+1]-1)
        j = J.rowval[ptr]
        j != i && (acc += J.nzval[ptr] * spins[j])
    end
    return acc
end

@inline delta(t::PairInteractionMatrix, spins, i, s_new) =
    _dot(s_new - (@inbounds spins[i]), partner_sum(t.J, spins, i))
@inline delta_energy(::PairInteractionMatrix, δ) = -δ
@inline commit!(::PairInteractionMatrix{TJ,AsymmetricCoupling}, δ) where TJ = nothing

energy(t::PairInteractionMatrix{TJ,SymmetricCoupling}) where TJ = -t.cache.val
energy(t::PairInteractionMatrix{TJ,AsymmetricCoupling}) where TJ =
    error("asymmetric J_ij: no Hamiltonian, cache deactivated")

function recompute!(t::PairInteractionMatrix{TJ,SymmetricCoupling}, spins) where TJ
    acc = 0.0
    @inbounds for i in eachindex(spins)
        acc += _dot(spins[i], partner_sum(t.J, spins, i))
    end
    t.cache.val = acc / 2
    return nothing
end
recompute!(t::PairInteractionMatrix{TJ,AsymmetricCoupling}, spins) where TJ = nothing

partners(t::PairInteractionMatrix, i) =
    Iterators.filter(!=(i), @view t.J.rowval[t.J.colptr[i]:(t.J.colptr[i+1]-1)])

# Two-site change (Kawasaki): cross-term looks up J_ij explicitly.
@inline function delta(t::PairInteractionMatrix{TJ,SymmetricCoupling}, spins, ij::NTuple{2,Int},
                       s_new::NTuple{2}) where TJ
    i, j = ij
    @inbounds dsi = s_new[1] - spins[i]
    @inbounds dsj = s_new[2] - spins[j]
    δ = _dot(dsi, partner_sum(t.J, spins, i)) + _dot(dsj, partner_sum(t.J, spins, j))
    Jij = zero(TJ)
    @inbounds for ptr in t.J.colptr[i]:(t.J.colptr[i+1]-1)
        t.J.rowval[ptr] == j && (Jij += t.J.nzval[ptr])
    end
    return δ + Jij * _dot(dsi, dsj)
end

delta(::PairInteractionMatrix{TJ,AsymmetricCoupling}, spins, ::NTuple{2,Int}, ::NTuple{2}) where TJ =
    error("two-site exchange: asymmetric J_ij has no Hamiltonian")
