module Metropolis
using Random

"""
    accept(log_weight::Function, args_new::Tuple{Number, N}, args_old::Tuple{Number, N}, rng::AbstractRNG)::Bool where N

Evaluate acceptance probability according to Metropolis criterium for imporance sampling of ``P(E) \\propto e^{log\\_weight(E)}``.

# Arguments
- `log_weight(args)`: logarithmic ensemble weight function, e.g., canomical ensemble ``log\\_weight(E) = -\\beta E``
- `args_new`: arguments (can be Number or Tuple) for new (proposed) state
- `args_old`: arguments (can be Number or Tuple) for old            state
- `rng`: random number generator, e.g. MersenneTwister

"""
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

#TODO: FunctionWrapper.jl
"""
    sweep(list_updates, list_probabilities::Vector{T}, rng::AbstractRNG; number_updates::Int=1) where T<:AbstractFloat

Randomly pick und run update (has to check acceptance by itself!) from
`list_updates` with probability specified in `list_probabilities` and repeat
this `number_updates` times.
"""
function sweep(list_updates, list_probabilities::Vector{T}, rng::AbstractRNG; number_updates::Int=1) where T<:AbstractFloat
  @assert length(list_updates) == length(list_probabilities)
  @assert (sum(list_probabilities) - 1.0) < 1e-6

  for i in 1:number_updates
    id = random_element(list_probabilities,rng)
    # update is requred to call Metropolis.accept() itself
    list_updates[id]()
  end
end

#TODO: Should we keep this specialization?
function sweep(update::Function, rng::AbstractRNG; number_updates::Int=1)
  for i in 1:number_updates
    # update is requred to call Metropolis.accept() itself
    update()
  end
end

"""
    random_element(list_probabilities::Vector{T},rng::AbstractRNG)::Int where T<:AbstractFloat

Pick an index from a list of probabilities.
"""
function random_element(list_probabilities::Vector{T},rng::AbstractRNG)::Int where T<:AbstractFloat
  @assert (sum(list_probabilities) - 1.0) < 1e-6
  theta = rand(rng)

  id = 1
  cumulated_prob = list_probabilities[id]
  while cumulated_prob < theta
    id += 1
    cumulated_prob += list_probabilities[id]
  end

  return id
end


end
export Metropolis
