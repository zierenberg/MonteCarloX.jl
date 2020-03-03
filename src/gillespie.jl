# Gillespie simulations wrapper 
struct Gillespie end

"""
    advance(alg::Gillespie, [rng::AbstractRNG], event_handler::AbstractEventHandlerRate, update!::Function, totol_time::T)::T where {T<:Real} 

Draw as many events for a `system` with a list of reactions with different
rates such that the time of the last event is larger than `total_time` and
return time of last event.
"""
function advance!(alg::Gillespie, rng::AbstractRNG, event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T<:AbstractFloat}
  alg = KineticMonteCarlo()
  time::T = 0
  while time <= total_time
    if length(event_handler) == 0
      println("WARNING: no events left before total_time reached")
      return time
    end
    dt, event = next(alg, rng, event_handler)
    time += dt
    update!(event_handler, event)
  end
  return time 
end
