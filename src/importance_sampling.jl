# importance sampling

struct Metropolis end
struct MetropolisHastings end #todo
struct HeatBath end #todo
struct Glauber end #todo

@doc """
    accept(log_weight::Function, args_new::Tuple{Number, N}, args_old::Tuple{Number, N}, rng::AbstractRNG)::Bool where N

Evaluate most general acceptance probability for imporance sampling of ``P(x) \\propto e^{log\\_weight(x)}``.

# Arguments
- `log_weight(args)`: logarithmic ensemble weight function, e.g., canomical ensemble ``log\\_weight(x) = -\\beta x``
- `args_new`: arguments (can be Number or Tuple) for new (proposed) state
- `args_old`: arguments (can be Number or Tuple) for old            state
- `rng`: random number generator, e.g. MersenneTwister

# Specializations
- accept(alg::Metropolis(), rng::AbstractRNG, x_new::T, x_old::T) where T (@ref)
"""
function accept(rng::AbstractRNG, log_weight::Function, args_new::NTuple{N,T}, args_old::NTuple{N,T})::Bool where {N,T}
  difference = log_weight(args_new...) - log_weight(args_old...)
  if difference > 0
    return true
  elseif rand(rng) < exp(difference)
    return true
  else
    return false
  end
end

function accept(rng::AbstractRNG, log_weight::Function, args_new::T, args_old::T)::Bool where T
  difference = log_weight(args_new) - log_weight(args_old)
  if difference > 0
    return true
  elseif rand(rng) < exp(difference)
    return true
  else
    return false
  end
end

@doc """
    accept(alg::Metropolis, rng::AbstractRNG, beta::Float64, dx::T)::Bool where T

Standard metropolis algorithm with 
```p(x\\to x^') = \\text{min}\\left(1, e^{-\\beta \\Delta x}\\right)```
where `dx = x^' - x`
"""
function accept(alg::Metropolis, rng::AbstractRNG, beta::Float64, dx::T)::Bool where T
  if dx < 0 
    return true
  elseif rand(rng) < exp(-beta*dx)
    return true
  else
    return false
  end
end

"""
    accept(alg::MetropolisHastings, rng::AbstractRNG, beta::Float64, dx::T, q_old::Float64, q_new::Float64)::Bool where T
"""
function accept(alg::MetropolisHastings, rng::AbstractRNG, beta::Float64, dx::T, q_old::Float64, q_new::Float64)::Bool where T
  println("not implemented yet because API uncleaer")
  return false
end


# sweep
@doc """
    sweep(list_updates, list_weights::AbstractWeights, rng::AbstractRNG; number_updates::Int=1) where T<:AbstractFloat

Randomly pick und run update (has to check acceptance by itself!) from
`list_updates` with probability specified in `list_probabilities` and repeat
this `number_updates` times.
"""
function sweep(list_updates, list_weights::AbstractWeights, rng::AbstractRNG; number_updates::Int=1)
  @assert length(list_updates) == length(list_weights)

  for i in 1:number_updates
    id = StatsBase.sample(rng, list_weights)
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
