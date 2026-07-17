# ── ExternalField: −Σ⟨h_i, σ_i⟩ with h in spin space ─────────────────────────
#
# Uniform h (Real or ComplexF64): the cached sum is Σσ, h is applied in delta_energy/energy.
# Site-dependent h (AbstractVector, random-field models): h_i cannot be factored out, so the
# cached sum is Σ⟨h_i, σ_i⟩ itself (exact Int for integer fields). Both share one struct,
# dispatch on the field type.

"""
    ExternalField(h)

Field term −Σ_i ⟨h, σ_i⟩; `h` a `Real`, a `ComplexF64` (in-plane XY field), or an
`AbstractVector` (site-dependent / random field).
"""
struct ExternalField{H, TC} <: AbstractInteraction
    h::H
    cache::Cache{TC}
end
ExternalField(h::Real) = ExternalField(h, Cache(0))
ExternalField(h::Complex) = ExternalField(ComplexF64(h), Cache(zero(ComplexF64)))
ExternalField(h::AbstractVector) = ExternalField(h, Cache(zero(promote_type(eltype(h), Int))))

@inline delta(t::ExternalField, spins, i, s_new) = _spin_sum_delta(s_new, @inbounds spins[i])
@inline delta_energy(t::ExternalField, δ) = -_dot(δ, t.h)
@inline energy(t::ExternalField) = -_dot(t.cache.val, t.h)
recompute!(t::ExternalField, spins) = (t.cache.val = _spin_sum(spins); nothing)

@inline delta(t::ExternalField{<:AbstractVector}, spins, i, s_new) =
    _dot(s_new - (@inbounds spins[i]), @inbounds t.h[i])
@inline delta_energy(::ExternalField{<:AbstractVector}, δ) = -δ
@inline energy(t::ExternalField{<:AbstractVector}) = -t.cache.val
recompute!(t::ExternalField{<:AbstractVector}, spins) =
    (t.cache.val = sum(i -> _dot(spins[i], t.h[i]), eachindex(spins)); nothing)

# Two-site change (Kawasaki).
@inline delta(t::ExternalField, spins, ij::NTuple{2,Int}, s_new::NTuple{2}) =
    _spin_sum_delta(s_new[1], @inbounds spins[ij[1]]) + _spin_sum_delta(s_new[2], @inbounds spins[ij[2]])
@inline delta(t::ExternalField{<:AbstractVector}, spins, ij::NTuple{2,Int}, s_new::NTuple{2}) =
    _dot(s_new[1] - (@inbounds spins[ij[1]]), @inbounds t.h[ij[1]]) +
    _dot(s_new[2] - (@inbounds spins[ij[2]]), @inbounds t.h[ij[2]])
