"""
# MonteCarloX.Gillespie

Module that allows to implement Gillespie algorithm for the statistical time trajectory of a stochastic equation
Uses kinetic Monte Carlo
"""
module Gillespie
using Random
using Distributions
using DataStructures
using ..KineticMonteCarlo

#TODO: ditch at some point
#abstract type AbstractSystem end
#function update!(system::AbstractSystem, id::Int) end
#function list_rates(system::AbstractSystem) end
#function sum_rates(system::AbstractSystem) end

using ..MonteCarloX
using ..EventHandler


"""
    advance(T::Ttime, list_rates::Vector{Trates}, update::Function, rng::AbstractRNG)::Ttime where {Ttime<:AbstractFloat, Trates<:AbstractFloat} 

Draw as many events for a `system` with a list of reactions with different
rates such that the time of the last event is larger than `total_time` and
return time of last event.
"""
#function advance!(system::AbstractSystem, total_time::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
#  time::T = 0
#  while time <= total_time
#    dt, id = KineticMonteCarlo.next_event(list_rates(system), sum_rates(system), rng)
#    time += dt
#    update!(system,id)
#  end
#  return time 
#end

function advance!(event_handler::AbstractEventHandler, update!::Function, total_time::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
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


###############################################################################
###############################################################################
###############################################################################
#mutable struct EventQueue{T}<:AbstractEventHandler
#  pq_event_rate::PriorityQueue{T,Float64} 
#  noevent::T
#  function EventQueue{T}(list_event::Vector{T}, list_time::Vector{Float64}, noevent::T) where T
#    @assert length(list_event)==length(list_time)
#    pq = PriorityQueue{T,Float64}()
#    for i=1:length(list_event)
#      enqueue!(pq,list_event[i]=>list_time[i])
#    end
#    new(pq, noevent)
#  end
#end
#
#function num_events(event_handler::EventQueue{T}) where T
#  return length(event_handler.pq_event_rate)
#end
#
#function set!(event_handler::EventQueue, event::T, time::Float64) where T
#  event_handler.pq_event_rate[event] = time 
#end
#
#function advance!(pq::PriorityQueue{T}, update!::Function, total_time::T, rng::AbstractRNG)::T where {T<:AbstractFloat}
#  time::T = 0
#  while time <= total_time
#    if lengthj(event_handler) == 0
#      println("WARNING: no events left before total_time reached")
#      return time
#    end
#    event,time = dequeue_pair!(event_handler.pq_event_rate) 
#    update!(event_handler, event)
#  end
#  return time 
#end


end
export Gillespie
