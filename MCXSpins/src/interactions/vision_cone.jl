# ── VisionConeInteraction: nonreciprocal pair term ────────────────────────────
#
# Direction is stored EXPLICITLY via the oriented lattice tables from systems/geometries.jl
# (`partners_pos[i]` = +axis neighbors: right/up/…; `partners_neg[i]` = −axis: left/down/…).

"""
    VisionConeInteraction(κ, dims::NTuple)

Nonreciprocal cone term (Garcés–Levis): every spin couples with an EXTRA `κ` to the
neighbors inside its vision cone — `partners_pos[i]` for an up spin, `partners_neg[i]` for
a down spin. The term carries ONLY the nonreciprocal extra: compose it with a
`PairInteraction(J, partners)` for the reciprocal coupling to all neighbors (see
`VisionConeIsingSystem`). The coupling is asymmetric (i may see j while j looks away):
there is NO Hamiltonian and NO cache. `delta_energy` is the dynamical local rule (cone
frozen at the current spin), not the gradient of any energy.
"""
struct VisionConeInteraction{D} <: AbstractInteraction
    κ::Float64
    partners_pos::Vector{NTuple{D,Int}}
    partners_neg::Vector{NTuple{D,Int}}
end
function VisionConeInteraction(κ::Real, dims::NTuple{D,<:Integer}) where D
    pos, neg = oriented_partners(Int.(dims))
    return VisionConeInteraction(Float64(κ), pos, neg)
end

symmetry(::VisionConeInteraction) = AsymmetricCoupling()

# Cache-free: the payload IS the dynamical ΔE.
@inline function delta(t::VisionConeInteraction, spins, i, s_new::Int8)
    @inbounds s_old = Int(spins[i])
    fwd = s_old > 0 ? partner_sum(t.partners_pos, spins, i) :
          (s_old < 0 ? partner_sum(t.partners_neg, spins, i) : 0)
    return -(Int(s_new) - s_old) * t.κ * fwd
end
@inline delta_energy(::VisionConeInteraction, δ) = δ
@inline commit!(::VisionConeInteraction, δ) = nothing
recompute!(::VisionConeInteraction, spins) = nothing
energy(::VisionConeInteraction) = error("vision-cone coupling: no Hamiltonian, cache-free")

partners(t::VisionConeInteraction, i) = (t.partners_pos[i]..., t.partners_neg[i]...)

delta(::VisionConeInteraction, spins, ::NTuple{2,Int}, ::NTuple{2}) =
    error("two-site exchange: not defined for the vision-cone dynamics")
