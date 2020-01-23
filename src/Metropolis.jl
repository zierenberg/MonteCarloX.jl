module Metropolis
using Random

#TODO: formulate Metropolis and everything in terms of logarithmic weights ... because it is also not directly P that we are passing (not normalized)
#accpect(log_weight, ...)

#TODO: Try general types (abstract) e.g. Number, Integer, float
# -> rethink Union in this cas
# -> Tuple{T} ... where T<:derived from sth :)

"""
Metropolis update probabilities of new and old states taking multiple arguments defined by model

this could also include the selection bias (Hastings)

# Arguments
- `log_weight(args)`: logarithmic ensemble weight function, e.g., canomical ensemble log_weight(E) = -\beta E
- `args_new`: tuple of arguments that are required by log_weight function for new (proposed) state
- `args_old`: tuple of arguments that are required by log_weight function for old state
- `rng`: random number generator, e.g. MersenneTwister

"""
#TODO type-stable function argument?
function accept(log_weight::Function, args_new::Number, args_old::Number, rng::AbstractRNG)::Bool
  difference = log_weight(args_new) - log_weight(args_old)
  if difference > 0
    return true
  elseif rand(rng) < exp(difference)
    return true
  else
    return false
  end
end

#TODO: check if syntax corrext or if dims need to be passes with where
function accept(log_weight::Function, args_new::Tuple{Number, N}, args_old::Tuple{Number, N}, rng::AbstractRNG)::Bool where N
  difference = log_weight(args_new...) - log_weight(args_old...)
  if difference > 0
    return true
  elseif rand(rng) < exp(difference)
    return true
  else
    return false
  end
end


#USELESS with the new definition in terms of log_weights :)
#"""
#Metropolis update probabilities of new and old states taking multiple arguments defined by model
#
#PLEASE BE CAREFUL AND KNOW WHAT YOU DO
#"""
#function accept_diff(log_weight, args_new, args_old,rng)
#  args_diff = args_new .- args_old
#  if rand(rng) < exp(log_weigth(args_diff...))
#    return true
#  else
#    return false
#  end
#end


"""
sweep that randomly picks update from a list according to probability

any recording has to be done on function argument level
"""
#TODO: work with cumulated_probabilities? -> should save time in random_element evaluation
# make speed check for small lists!!!
# FunctionWrapper.jl
function sweep(list_updates, list_probabilities::Vector{Float64}, rng::AbstractRNG; number_updates::Int=1)
  @assert length(list_updates) == length(list_probabilities)
  @assert sum(list_probabilities) == 1
#  @assert cumulated_probabilities[end] == 1

  for i in 1:number_updates
    id = random_element(list_probabilities,rng)
    #id = binary_search(cumulated_probabilities, rand(rng))
    #calls metropolis itself
    list_updates[id]()
  end
end

"""
sweep that calls a single function

TODO: Should we keep this functionality?
"""
function sweep(update, rng::AbstractRNG; number_updates::Int=1)
  for i in 1:number_updates
    #calls metropolis itself
    update()
  end
end

"""
generate events (id) with given probability

for optiomal performance have a sorted list with highest probabilities first

This should actually be a bit faster than binary search especially if stick to convention to start with high probabilities!!!

TODO: find out why random element is actually slower after second call of
function test_sweep_random_element(;verbose=false)
(first call clearly faster than binary search)

"""
function random_element(list_probs::Vector{Float64},rng::AbstractRNG)
  theta = rand(rng)

  id = 1
  cumulated_prob = list_probs[id]
  while cumulated_prob < theta
    id += 1
    cumulated_prob += list_probs[id]
  end

  return id
end


end
export Metropolis
