# ── CrystalField: +Δ Σσ² (discrete spins) ─────────────────────────────────────

"""
    CrystalField(Δ)

Crystal-field term +Δ Σ_i σ_i² (discrete spins). Caches Σσ² (before Δ).
"""
struct CrystalField{TΔ<:Real} <: AbstractInteraction
    Δ::TΔ
    cache::Cache{Int}
end
CrystalField(Δ::Real) = CrystalField(Δ, Cache(0))

@inline delta(t::CrystalField, spins, i, s_new::Int8) =
    Int(s_new)^2 - Int(@inbounds spins[i])^2
@inline delta_energy(t::CrystalField, δ) = t.Δ * δ
@inline energy(t::CrystalField) = t.Δ * t.cache.val

recompute!(t::CrystalField, spins) = (t.cache.val = sum(s -> Int(s)^2, spins); nothing)

# Two-site change (Kawasaki).
@inline delta(t::CrystalField, spins, ij::NTuple{2,Int}, s_new::NTuple{2,Int8}) =
    (Int(s_new[1])^2 - Int(@inbounds spins[ij[1]])^2) +
    (Int(s_new[2])^2 - Int(@inbounds spins[ij[2]])^2)
