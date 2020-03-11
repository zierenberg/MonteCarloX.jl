# Gillespie simulations wrapper 
struct Gillespie end

"""
#Remarks
We could think about a generalization of this to pass also custom events, threhold, etc. But then this exactly becomes the event_handler function ... So I would keep this as a specialization.

#See also direct construction of event handlers
[`ListEventRateSimple`](@ref), [`ListEventRateActiveMask`](@ref)
"""
function initialize(alg::Gillespie, rates::Vector{Float64}; type_event_handler::String = "ListEventRateSimple")
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

function next(alg::Gillespie, rng::AbstractRNG, event_handler::AbstractEventHandlerRate)::Tuple{Float64,Int}
  next(KineticMonteCarlo(), rng, event_handler)
end

function advance!(alg::Gillespie, rng::AbstractRNG, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat}
  advance!(KineticMonteCarlo(), rng, event_handler, update!, total_time)
end

function advance!(alg::Gillespie, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T <: AbstractFloat} 
  advance!(alg, Random.GLOBAL_RNG, event_handler, update!, total_time)
end
