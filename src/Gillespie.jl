"""
# MonteCarloX.Gillespie

Module that allows to implement Gillespie algorithm for the statistical time trajectory of a stochastic equation
Uses kinetic Monte Carlo
"""
module Gillespie
using Random
using Distributions

function step(list_updates, list_rates, rng)
  @assert length(list_updates) == length(list_rates)
  dt, id = next_event_rate(list_rates, rng)
  list_updates[id]()
  return dt
end

"""
Advance the system by time T 

The function generates new events as long as the total time increment does not
go beyond the total time T. If the next event would advance the time too far,
then this event is NOT performed. The system is thus in a state that it would
have after time T. 

Assuming Poisson rates, the function can be called again to advance from time T
because generation of events is independent.
(TODO: check)
"""
function advance(T, list_updates, list_rates, rng)
  @assert length(list_updates) == length(list_rates)
  dT = 0
  while dT < T
    dt, id = next_event_rate(list_rates, rng)
    if dT + dt < T
      list_updates[id]()
      dT += dt
    else
      return
    end
  end
  return 
end


#For convenience? (speed?)
"""
generate events (dt,id) from a list of rates such that their occurence corresponds with their rate
"""
function next_event_rate(list_rates::Array,rng::AbstractRNG)
  return KineticMonteCarlo.next_event_rate(list_rates,rng)
end

"""
fast implementation of next_event_rate if sum(list_rates) is known
"""
function next_event_rate(list_rates::Array,sum_rates::Float64,rng::AbstractRNG)
  return KineticMonteCarlo.next_event_rate(list_rates, sum_rates, rng)
end

#TODO: Gillespie to "manage" list of events?
# I imagine somehting like passing list of updates and list of rates and to then perform the particular one that is brought up by next_event
#
# potentially also in terms of having populations that all have the same rates (resort and keep updated) and where then only one of a couple of rates is drawn randomly
#
# In other case, also keep the list of rates updated? Then one would need a map, or a function that is called upon accepting a certain rate...
#


end
export Gillespie
