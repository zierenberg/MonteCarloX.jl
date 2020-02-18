"""
Module for timeseries and histogram reweighting of observables sampled in equilibrium
"""
module Reweighting
using Distributions
using ..MonteCarloX

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
  - Dictionary
"""
function distribution_from_timeseries(log_P_target, log_P_source, list_args)
  N = length(list_args)
  list_log_weight_ratio = [log_P_target(list_args[i]...)-log_P_source(list_args[i]...) for i in 1:N]
  
  log_norm = -Inf
  for i in 1:N
    log_norm  = MonteCarloX.log_sum(log_norm, list_log_weight_ratio[i])
  end
  distribution=Dict{typeof(list_args[1]),Float64}() 
  for i in 1:N
    Histograms.add(distribution, list_args[i], increment = exp(list_log_weight_ratio[i] - log_norm))
  end
  Histograms.normalize!(distribution)
  return distribution
end


"""
Estimate expectation value from histogram (

Ref: Janke

Todo: rename log_P... to log_weight!!

important: hist_obs(args) = sum O_i delta(args - args_i)
hists are dictionaries?
can this be generalized to higher dimensions? nd histograms as dictionary?
"""
function expectation_value_from_histogram(f_args::Function, log_P_target::Function, log_P_source::Function, hist::Dict)
  log_norm = log_normalization(log_P_target, log_P_source, hist)
  expectation_value = 0 
  for (args,H) in hist
    expectation_value += f_args(args)*H*exp(log_P_target(args...) - log_P_source(args...)-log_norm)
  end
  return expectation_value
end


function expectation_value_from_histogram(log_P_target::Function, log_P_source::Function, hist::Dict, hist_obs::Dict)
  log_norm = log_normalization(log_P_target, log_P_source, hist)
  expectation_value = 0 
  for (args,sum_obs) in hist_obs
    expectation_value += sum_obs*exp(log_P_target(args...) - log_P_source(args...) - log_norm)
  end
  return expectation_value
end

function log_normalization(log_P_target, log_P_source, hist::Dict)
  log_norm = -Inf
  for (args,H) in hist
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

end
export Reweighting
