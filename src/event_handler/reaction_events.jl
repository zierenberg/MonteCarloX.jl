# ── ReactionEvents: the reaction-network event generator ──────────────────────
#
# Second member of the generator family (see site_events.jl): events are the REACTION
# CHANNELS of a well-mixed network — association, dissociation, birth, decay, … — with
# state-dependent propensities from a rate rule. The classic Gillespie SSA setting.
#
# Networks have few channels, so after each event ALL propensities are refreshed (O(R) per
# event). A species→reaction dependency graph, refreshing only the channels that share
# species with the fired one, is the natural optimization if large networks ever appear.
#
# A model plugs in through the reaction interface (systems subtype `AbstractSystem`):
#
#     nreactions(sys)          — number of reaction channels
#     modify!(sys, r, t)       — fire channel r (update the species counts)
#
# and the rate rule is any callable `(sys, r) -> propensity`, e.g. mass action written out.

"""
    nreactions(sys)

Number of reaction channels of a well-mixed network system. Extension point for
[`ReactionEvents`](@ref).

API: This still needs to be implemented by the user for their system type, maybe we can find a more natural solution.
"""
function nreactions end

"""
    ReactionEvents(sys, rates)

Event generator over the reaction channels of `sys`: events are channel indices `1:R` with
propensities `rates(sys, r)` — mass action written as a plain callable. Feed to the KMC
loop like any event source:

    src = ReactionEvents(sys, (s, r) -> r == 1 ? s.k_on * s.A * s.B : s.k_off * s.AB)
    advance!(Gillespie(rng), src, total_time; observe!)

The generator maintains its propensities itself: its `modify!` fires the channel on `sys`
and re-evaluates all propensities. If you modify `sys` outside the loop, reconstruct the
generator (O(R)).

Note: This is still completely experimental
"""
struct ReactionEvents{S,R} <: AbstractEventHandlerRate{Int}
    sys::S
    rates::R
    tree::EventRateTree
end

function ReactionEvents(sys, rates)
    re = ReactionEvents(sys, rates, EventRateTree(nreactions(sys)))
    _refresh_reactions!(re)
    @warn "ReactionEvents is still experimental; the API may change."
    return re
end

function _refresh_reactions!(re::ReactionEvents)
    for r in 1:length(re.tree)
        re.tree[r] = re.rates(re.sys, r)
    end
    return nothing
end

Base.length(re::ReactionEvents) = length(re.tree)
total_rate(re::ReactionEvents) = total_rate(re.tree)
next_event(rng::AbstractRNG, re::ReactionEvents) = next_event(rng, re.tree)

# Fire the drawn channel on the system and re-evaluate the propensities.
function modify!(re::ReactionEvents, event::Int, t)
    modify!(re.sys, event, t)
    _refresh_reactions!(re)
    return nothing
end
