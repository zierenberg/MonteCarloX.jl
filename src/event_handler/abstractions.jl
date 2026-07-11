# ── Event handlers ────────────────────────────────────────────────────────────
#
# Containers feeding the kinetic Monte Carlo loop. Two families:
#
#   AbstractEventHandlerRate{T} — events with RATES; the KMC step draws a Poissonian waiting
#       time from `total_rate` and an event of type T from `next_event`. Members:
#         ListEventRateSimple     O(N) linear scan; the honest baseline for small/static N
#         ListEventRateActiveMask O(N) scan over ACTIVE events only; for models where events
#                                 switch on/off (e.g. contact processes)
#         EventRateTree           O(log N) Fenwick tree; the scaling path when events change
#                                 a few rates per step (n-fold way)
#   AbstractEventHandlerTime{T} — events with SCHEDULED TIMES (EventQueue); the KMC step pops
#       the earliest event and advances the clock to it.
#
# Contract for a new rate handler: subtype AbstractEventHandlerRate{T} and provide either
#   total_rate(h), next_event(rng, h), length(h), h.noevent      (own sampling — cf. the tree)
# or the index protocol the generic scan in kinetic_monte_carlo.jl walks:
#   first_index(h), last_index(h), next_index(h, i), previous_index(h, i),
#   h.list_rate, length(h), h.noevent                            (cf. ListEventRateActiveMask)

abstract type AbstractEventHandler{T} end
abstract type AbstractEventHandlerTime{T} <: AbstractEventHandler{T} end
abstract type AbstractEventHandlerRate{T} <: AbstractEventHandler{T} end

function Base.getindex(event_handler::AbstractEventHandlerRate, index::Int64)
    return event_handler.list_rate[index]
end
