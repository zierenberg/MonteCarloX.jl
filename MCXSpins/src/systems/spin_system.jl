# ── SpinSystem: a spin type + a tuple of interactions ─────────────────────────
#
# A system is
#
#     SpinSystem(spintype, interactions::Tuple)
#
# where `spintype` fixes the local degree of freedom (value set, storage, proposal — see
# spin_types.jl) and each interaction owns its couplings, index structure, and cache (see
# interactions/). The system plugs into the generic spin_flip! via the standard hooks
# (propose_state → delta_sys → delta_energy → accept! → modify!).

"""
    SpinSystem(spintype, interactions::Tuple; geometry=nothing)

Spin system composed of a spin type and a tuple of interaction terms, e.g.

    SpinSystem(Spin(1), (PairInteraction(J, partners), CrystalField(Δ)))

Starts in the all-up reference state; use [`init!`](@ref) or [`set_spins!`](@ref). The
struct is immutable: the mutable pieces (interaction caches, running spin sum) live in
`Cache` cells. `spin_sum` is Σσ — it belongs to the spin vector, not to any interaction:
Int (exact) for discrete spins, ComplexF64 for XY, `SVector{3,Float64}` for Heisenberg.
"""
struct SpinSystem{V, S<:SpinType, I<:Tuple, M, G} <: AbstractSpinSystem
    spins::Vector{V}
    spintype::S
    interactions::I
    spin_sum::Cache{M}
    geometry::G      # PASSIVE metadata (dims tuple, graph, or nothing) for observables and
                     # parallel scheduling — the dynamics never reads it; topology used by the
                     # dynamics lives inside the interactions
end

function SpinSystem(spintype::SpinType, interactions::Tuple, N::Integer; geometry=nothing)
    spins = fill(one_state(spintype), N)
    return recompute_all!(SpinSystem(spins, spintype, interactions,
                                     Cache(_spin_sum(spins)), geometry))
end

# ── Hooks for the generic spin_flip! ──────────────────────────────────────────
#
# The delta payload δs holds one delta per interaction term, computed ONCE in delta_sys and
# consumed by delta_energy (accept decision) and the four-argument modify! (commit to the
# caches). The four-argument form is called only by the system-level updates; the
# three-argument modify!(sys, i, s_new) below remains the raw-state entry point (heat bath,
# n-fold way, tests) and computes the payload itself.

# Fetch the current state at site i and hand it to the spin type's proposal. Any extra proposal
# parameter (e.g. the XY rotation half-width Δθ) passes straight through.
@inline propose_state(rng::AbstractRNG, sys::SpinSystem, i, params...) =
    propose_state(rng, sys.spintype, (@inbounds sys.spins[i]), params...)

@inline delta_sys(sys::SpinSystem, i, s_new) = delta(sys.interactions, sys.spins, i, s_new)

@inline delta_energy(sys::SpinSystem, i, s_new, δs::Tuple) = delta_energy(sys.interactions, δs)
@inline delta_energy(sys::SpinSystem, i, s_new) = delta_energy(sys, i, s_new, delta_sys(sys, i, s_new))

@inline function MonteCarloX.modify!(sys::SpinSystem, i::Int, s_new, δs::Tuple)
    commit!(sys.interactions, δs)
    @inbounds s_old = sys.spins[i]
    @inbounds sys.spins[i] = s_new
    sys.spin_sum.val += _spin_sum_delta(s_new, s_old)
    return nothing
end
# Raw-state entry point (heat bath, n-fold way, tests, external drivers): build the payload.
MonteCarloX.modify!(sys::SpinSystem, i::Int, s_new) = modify!(sys, i, s_new, delta_sys(sys, i, s_new))

# ── Observables and bookkeeping ───────────────────────────────────────────────

"""
    energy(sys::SpinSystem; full=false)

Total energy: strictly O(1) from the interaction caches; throws if any term is cache-free
(no Hamiltonian — see `hamiltonian_energy`). `full=true` rebuilds all caches first (O(N)
reference path).
"""
@inline energy(sys::SpinSystem; full=false) =
    (full && recompute_all!(sys); energy(sys.interactions))

"""
    magnetization(sys::SpinSystem; full=false)

Total spin sum Σσ from the running cache (Int for discrete, ComplexF64 for XY,
`SVector{3,Float64}` for Heisenberg). `full=true` recomputes from the spins.
"""
@inline magnetization(sys::SpinSystem; full=false) =
    full ? _spin_sum(sys.spins) : sys.spin_sum.val

"""
    hamiltonian_energy(sys::SpinSystem)

Energy of the symmetric interactions only — the proper Hamiltonian part of the system. This
is the quantity multicanonical / reweighting methods may legitimately use; for a fully
symmetric system it equals `energy(sys)`.
"""
hamiltonian_energy(sys::SpinSystem) = hamiltonian_energy(sys.interactions)

"""
    is_hamiltonian(sys::SpinSystem)

Whether every interaction derives from a proper Hamiltonian (no asymmetric couplings).
"""
is_hamiltonian(sys::SpinSystem) = all(t -> symmetry(t) isa SymmetricCoupling, sys.interactions)

# O(N) reference path: rebuild the running spin sum and every interaction cache from the
# spins. Called at construction, by set_spins!/init!, and after cluster moves.
function recompute_all!(sys::SpinSystem)
    sys.spin_sum.val = _spin_sum(sys.spins)
    foreach(t -> recompute!(t, sys.spins), sys.interactions)
    return sys
end

"""
    set_spins!(sys::SpinSystem, spins)

Overwrite the configuration and rebuild all caches.
"""
function set_spins!(sys::SpinSystem, spins)
    sys.spins .= spins
    return recompute_all!(sys)
end

"""
    init!(sys::SpinSystem, type; rng=nothing)

Initialize the configuration: `:up` (all `one_state`), `:down` (all `-one_state`), `:zero`
(all σ = 0; discrete spin types containing 0 only), or `:random` (uniform draws, requires
`rng`). Rebuilds all caches.
"""
function init!(sys::SpinSystem, type::Symbol; rng=nothing)
    if type == :up
        fill!(sys.spins, one_state(sys.spintype))
    elseif type == :down
        fill!(sys.spins, -one_state(sys.spintype))
    elseif type == :zero
        sys.spintype isa Spin && Int8(0) in states(sys.spintype) ||
            error("Initialization :zero requires a discrete spin type containing σ = 0")
        fill!(sys.spins, Int8(0))
    elseif type == :random
        @assert rng !== nothing "Random initialization requires rng"
        @inbounds for i in eachindex(sys.spins)
            sys.spins[i] = random_state(rng, sys.spintype)
        end
    else
        error("Unknown initialization type: $type")
    end
    return recompute_all!(sys)
end

# ── Geometry queries (setup/observable utilities — not hot-path) ──────────────

partners(sys::SpinSystem, i) = partners(sys.interactions, i)

# ── local-states interface (core SiteEvents/NFoldRates: nsites/local_states/delta_energy/modify!/partners) ──

MonteCarloX.nsites(sys::SpinSystem) = length(sys.spins)

"""
    local_states(sys::SpinSystem{<:Any, <:Spin}, i)

All target states of site `i` except its current one, as a compile-time tuple (the
skip-current mapping: slot `a` holds `states[a]`, a collision with the current state maps
to the last state — each alternative appears exactly once).
"""
@inline MonteCarloX.local_states(sys::SpinSystem{<:Any, <:Spin}, i) =
    _local_states(states(sys.spintype), @inbounds sys.spins[i])
@inline _local_states(sts::NTuple{N, Int8}, s::Int8) where N =
    ntuple(a -> (@inbounds sts[a] == s ? sts[N] : sts[a]), Val(N - 1))
MonteCarloX.local_states(sys::SpinSystem, i) =
    error("local-transition events require a discrete spin type (Spin{S})")

"""
    geometry(sys)

Passive geometry metadata attached at construction: lattice dimensions (`NTuple`), a
`SimpleGraph`, or `nothing` (e.g. bare coupling matrices). For observables (structure
factor, correlation length) and parallel scheduling; the dynamics never reads it.
"""
geometry(sys::SpinSystem) = sys.geometry
