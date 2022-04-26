# Gillespie simulations wrapper
struct Gillespie end

"""
    init(rng, Gillespie(), rates)

create a simple KineticMonteCarlo simulation object that handles Gillespie-type
simulation.

[`SimulationKineticMonteCarlo`](@ref)
"""
function init(
        rng::AbstractRNG,
        alg::Gillespie,
        rates::Vector{Float64},
        type_event_handler::String = "ListEventRateSimple"
    )
    events = collect(1:length(rates))
    noevent = 0
    threshold = 0.0
    if type_event_handler == "ListEventRateSimple"
     event_handler = ListEventRateSimple{Int}(events, rates, threshold, noevent)
    elseif type_event_handler == "ListEventRateActiveMask"
      event_handler = ListEventRateActiveMask{Int}(events, rates, threshold, noevent)
    else
      throw(UndefVarError(:type_event_handler))
    end

    return SimulationKineticMonteCarlo(rng, event_handler)
end
init(alg::Gillespie, rates, type_event_handler = "ListEventRateSimple") = init(Random.GLOBAL_RNG, alg, rates, type_event_handler)
