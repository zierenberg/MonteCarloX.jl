# ── SiteEvents: the local-transition event generator ─────────────────────────
#
# An event GENERATOR: an event handler that maintains its own rate ledger from a model
# interface, instead of being maintained by the user's modify! (the passive containers).
# First member of the generator family, covering every "sites change their local states"
# dynamics — the n-fold way on spin systems, contact processes, ecological lattice models:
#
#     events         (site i, target state s) for s in local_states(sys, i)
#     rates          rates(sys, i, s) — the pluggable RATE RULE (see below)
#     locality       applying an event invalidates only site i and partners(sys, i)
#
# Rates live in an EventRateTree (O(log N) update and sampling, never copied). The generator
# plugs into the ordinary KMC loop as any other event source: `next` draws `(dt, (i, s_new))`,
# and its `modify!` applies the move to the system and refreshes the invalidated rates.
#
# The rate rule is any callable `(sys, i, s_new) -> rate`:
#   • intrinsic dynamics — write the physics directly (infection/recovery rates, …);
#   • sampling — [`NFoldRates`](@ref)`(ensemble; balance)`: the Bortz–Kalos–Lebowitz n-fold
#     way, i.e. rejection-free {Metropolis|Glauber} dynamics. The n-fold way IS a Gillespie
#     algorithm: Gillespie's clock over balance-induced local-transition rates.
#
# A model plugs in through the SITE INTERFACE (src/abstract_system.jl): nsites,
# local_states, partners, modify!(sys, i, s), and — only for NFoldRates — delta_energy.

"""
    NFoldRates(ensemble; balance=MetropolisBalance())
    NFoldRates(; β, balance=MetropolisBalance())

Rate rule of the n-fold way (Bortz–Kalos–Lebowitz): the event `(i, s_new)` gets the rate

    transition_rate(balance, logweight(ensemble, ΔE))

— the same balance function the Metropolis/Glauber accept step applies in discrete time,
read as a continuous-time rate. Sampling with `SiteEvents(sys, NFoldRates(β=β))` under
`Gillespie` is therefore rejection-free {Metropolis|Glauber} dynamics: time averages equal
the ensemble averages of the embedded chain. Linear ensembles only.
"""
struct NFoldRates{E,B<:BalanceFunction}
    ensemble::E
    balance::B
    function NFoldRates(ensemble; balance::BalanceFunction=MetropolisBalance())
        assert_linear_ensemble(ensemble, "n-fold rates")
        e = _as_ensemble(ensemble)
        @warn "NFoldRates is still experimental; the API may change."
        return new{typeof(e),typeof(balance)}(e, balance)
    end
end
NFoldRates(; β::Real, balance::BalanceFunction=MetropolisBalance()) =
    NFoldRates(BoltzmannEnsemble(β=β); balance=balance)

@inline (rates::NFoldRates)(sys, i, s_new) =
    transition_rate(rates.balance, logweight(rates.ensemble, delta_energy(sys, i, s_new)))

"""
    SiteEvents(sys, rates)

Event generator over the local transitions of `sys`: events are `(i, s_new)` pairs with
`rates(sys, i, s_new)` from the rate rule — a plain callable for intrinsic dynamics, or
[`NFoldRates`](@ref) for rejection-free sampling. Feed to the KMC loop like any event
source:

    src = SiteEvents(sys, NFoldRates(β=β))
    advance!(Gillespie(rng), src, total_time; observe!)

The generator maintains its rates itself (in an [`EventRateTree`](@ref)): its `modify!`
applies the drawn move to `sys` and refreshes the rates of the site and its `partners`.
If you modify `sys` outside the loop, reconstruct the generator (O(N)).
"""
struct SiteEvents{S,R,V} <: AbstractEventHandlerRate{Tuple{Int,V}}
    sys::S
    rates::R
    tree::EventRateTree
    n_alt::Int
end

function SiteEvents(sys, rates)
    sts = local_states(sys, 1)
    n_alt = length(sts)
    le = SiteEvents{typeof(sys),typeof(rates),eltype(sts)}(
        sys, rates, EventRateTree(nsites(sys) * n_alt), n_alt)
    for i in 1:nsites(sys)
        _refresh_site!(le, i)
    end
    @warn "SiteEvents is still experimental; the API may change."
    return le
end

function _refresh_site!(le::SiteEvents, i::Int)
    base = (i - 1) * le.n_alt
    sts = local_states(le.sys, i)
    for a in 1:le.n_alt
        le.tree[base + a] = le.rates(le.sys, i, sts[a])
    end
    return nothing
end

Base.length(le::SiteEvents) = length(le.tree)
total_rate(le::SiteEvents) = total_rate(le.tree)

function next_event(rng::AbstractRNG, le::SiteEvents)
    event = next_event(rng, le.tree)
    i, a = fldmod1(event, le.n_alt)
    return (i, local_states(le.sys, i)[a])
end

# Apply a drawn event to the system and refresh the rates it invalidated.
function modify!(le::SiteEvents, (i, s_new)::Tuple, t)
    modify!(le.sys, i, s_new)
    _refresh_site!(le, i)
    for j in partners(le.sys, i)         # duplicates re-derive the same rates: harmless
        _refresh_site!(le, j)
    end
    return nothing
end
