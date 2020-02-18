"""
# MonteCarloX.Gillespie

Module that allows to implement Gillespie algorithm for the statistical time trajectory of a stochastic equation
Uses kinetic Monte Carlo
"""
module Gillespie
using Random
using Distributions
using DataStructures

#TODO: ditch at some point
#abstract type AbstractSystem end
#function update!(system::AbstractSystem, id::Int) end
#function list_rates(system::AbstractSystem) end
#function sum_rates(system::AbstractSystem) end

using ..MonteCarloX
using ..KineticMonteCarlo
using ..EventHandler


"""
    advance(T::Ttime, list_rates::Vector{Trates}, update::Function, rng::AbstractRNG)::Ttime where {Ttime<:AbstractFloat, Trates<:AbstractFloat} 

Draw as many events for a `system` with a list of reactions with different
rates such that the time of the last event is larger than `total_time` and
return time of last event.
"""

function advance!(event_handler::AbstractEventHandlerRate, update!::Function, total_time::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
  time::T = 0
  while time <= total_time
    if num_events(event_handler) == 0
      println("WARNING: no events left before total_time reached")
      return time
    end
    dt, event = KineticMonteCarlo.next_event(event_handler, rng)
    time += dt
    update!(event_handler, event)
  end
  return time 
end

end
export Gillespie
