#Reweighting functions
#TODO: change names so that reweight is included because no longer namespace Reweighting

"""
    expectation_value_from_timeseries_log(log_P_target::Function, log_P_source::Function, list_args, list_obs::Vector{Tin})::Tout where {Tin<:Number,Tout<:AbstractFloat}

Calculate the expectation value of an observable in `P_target` from a list of measured observables in `P_source`.

This can be used for observables measured in equilibrium, e.g., from methods:
- metropolis         (for each temperature separate)
- parallel tempering (for each temperature separate)
- multicanonical
- population annealing 

# Background
 
Definition of reweighting in general: 

``\\langle O\\rangle = \\sum O_i P_\\mathrm{target}(args_i)/P_\\mathrm{source}(args_i) / \\sum P_\\mathrm{target}(args_i)/P_\\mathrm{source}(args_i)``

Definition of reweighting for the canonical ensemble:

``\\langle O\\rangle = \\sum O_i e^{\\beta_\\mathrm{target} E_i - \\beta_\\mathrm{source} E_i} / \\sum e^{\\beta_\\mathrm{target} E_i - \\beta_\\mathrm{source} E_i}``

# Remark
So far, this may not be well implemented for type stability. However, it should not be the most timeconsuming part of the simulation so this problem is moved to later time.
"""
function expectation_value_from_timeseries(log_P_target::Function, log_P_source::Function, list_args, list_obs::Vector{Tin})::Float64 where {Tin<:Number}
  N = length(list_obs)
  @assert N == length(list_args)
  #function for difference between logarithmic weights instead of copying this
  #into an extra array. Unclear if this is better. 
  log_weight_diff(i) = log_P_target(list_args[i]...) - log_P_source(list_args[i]...)
  
  log_norm = -Inf
  for i in 1:N
    log_norm  = MonteCarloX.log_sum(log_norm, log_weight_diff(i))
  end

  expectation_value = 0 
  for i in 1:N
    expectation_value += list_obs[i]*exp(log_weight_diff(i) - log_norm)
  end
  return expectation_value
end


"""
Estimate distribution from a list of (measured) arguments to an (n-dimensional) probability distribution

For higher dimensional distributions (e.g. P(E,M)) list_args needs to be a list of tuples

returns:
  - StatsBase.Histogram

"""
#TODO: need control over bin size
#TODO: need to figure out how to generate histogram objects with float
function distribution_from_timeseries(log_P_target, log_P_source, list_args, range)
  N = length(list_args)
  list_weights = Weights([log_P_target(list_args[i]...)-log_P_source(list_args[i]...) for i=1:N])
  
  log_norm = -Inf
  for i in 1:N
    log_norm  = MonteCarloX.log_sum(log_norm, list_weights[i])
  end
  list_weights .= exp.(list_weights .- log_norm)
  hist = fit(Histogram, list_args, list_weights, range, closed=:left) 
  return normalize(hist, mode=:pdf)
  #Distribution=Dict{typeof(list_args[1]),Float64}() 
  #for i in 1:N
  #  add!(Distribution, list_args[i], increment = exp(list_log_weight_ratio[i] - log_norm))
  #end
  #normalize!(Distribution)
  #return Distribution
end


"""
Estimate expectation value from histogram (

Ref: Janke

Todo: rename log_P... to log_weight!!

important: hist_obs(args) = sum O_i delta(args - args_i)
hists are dictionaries?
can this be generalized to higher dimensions? nd histograms as dictionary?
"""
function expectation_value_from_histogram(f_args::Function, log_P_target::Function, log_P_source::Function, hist::Histogram)
  log_norm = log_normalization(log_P_target, log_P_source, hist)
  expectation_value = 0 
  for (args, H) in zip(hist.edges[1], hist.weights)
    expectation_value += f_args(args)*H*exp(log_P_target(args...) - log_P_source(args...)-log_norm)
  end
  return expectation_value
end

#TODO: this is only valid for 1D histograms!!
# what is missing in StatsBase is an API to iterate over histogram edges and weights (multidimensional)
function expectation_value_from_histogram(log_P_target::Function, log_P_source::Function, hist::Histogram, hist_obs::Histogram)
  log_norm = log_normalization(log_P_target, log_P_source, hist)
  expectation_value = 0 
  for (args, sum_obs) in zip(hist_obs.edges[1], hist_obs.weights)
    expectation_value += sum_obs*exp(log_P_target(args...) - log_P_source(args...) - log_norm)
  end
  return expectation_value
end

function log_normalization(log_P_target, log_P_source, hist::Histogram)
  log_norm = -Inf
  for (args, H) in zip(hist.edges[1], hist.weights)
    log_norm  = MonteCarloX.log_sum(log_norm, log(H) + log_P_target(args...) - log_P_source(args...))
  end
  return log_norm
end


###############################################################################
### Special wrapper for special ensembles

"""
timeseries reweighting in canonical ensemble

methods:
- metropolis         (for each temperature separate)

<O> = sum O_i P_target(E_i)/P_source(E_i) / sum P_target(E_i)/P_source(args_i)

"""

