# Gillespie simulations wrapper
"""
    Gillespie()

Spezializations for Gillespie-type simulations derived from KineticMonteCarlo().

# Basic functions
* [`next`](@ref)
* [`advance!`](@ref)

# Additional functions
* [`initialize`](@ref)
"""
struct Gillespie end

"""
    initialize(alg::Gillespie, rates)

creates `event handler` from a list of `rates` with most simple types of events
[1:length(rates)]. With `type_event_handler` one can specify the class of event
handler (default is `ListEventRateSimple`)

#See also direct construction of event handlers
[`ListEventRateSimple`](@ref), [`ListEventRateActiveMask`](@ref)
"""
function initialize(alg::Gillespie,
                    rates::Vector{Float64};
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

  return event_handler
end

function next(rng::AbstractRNG, alg::Gillespie, event_handler::AbstractEventHandlerRate)::Tuple{Float64,Int}
  next(rng, KineticMonteCarlo(), event_handler)
end

#should work already
#function advance!(rng::AbstractRNG, alg::Gillespie, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat}
#  advance!(rng, KineticMonteCarlo(), event_handler, update!, total_time)
#end
## If no rng is specified, uses Random.GLOBAL_RNG per default
#function advance!(alg::Gillespie, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat}
#  advance!(alg, event_handler, update!, total_time)
#end
