module Reweighting
using Distributions

include("Histogram.jl")
#function add(hist::Dict, args; value=1)
#  if args in keys(hist)
#    hist[args] += value
#  else
#    hist[args]  = value
#  end
#end

#TODO: move to helper class
"""
logarithmic addition of type C = e^c = A+B = e^a + e^b
 c = ln(A+B) = a + ln(1+e^{b-a})
 with b-a < 1
"""
function log_sum(a,b)
  if b < a
    return a + log(1+exp(b-a)) 
  else
    return b + log(1+exp(a-b))
  end
end

function normalize!(dist::Dict)
  norm = 0.0
  for (args, P) in dist
    norm += P
  end
  for (args, P) in dist
    dist[args] = P/norm
  end
end

"""
timeseries reweighting 

can be used with methods:
- metropolis         (for each temperature separate)
- parallel tempering (for each temperature separate)
- multicanonical
- population annealing?

canonical
<O> = sum O_i P_target(E_i)/P_source(E_i) / sum P_target(E_i)/P_source(args_i)

or in general
<O> = sum O_i P_target(args_i)/P_source(args_i) / sum P_target(args_i)/P_source(args_i)
"""
function expectation_value_from_timeseries_log(log_P_target, log_P_source, list_args, list_obs)
  N = length(list_obs)
  @assert length(list_obs)==length(list_args)
  list_log_weight_ratio = [log_P_target(list_args[i]...)-log_P_source(list_args[i]...) for i in 1:N]
  
  log_norm = -Inf
  for i in 1:N
    log_norm  = log_sum(log_norm, list_log_weight_ratio[i])
  end
  expectation_value = 0 
  for i in 1:N
    expectation_value += list_obs[i]*exp(list_log_weight_ratio[i] - log_norm)
  end
  return expectation_value
end


"""
timeseries reweighting 

wrapper for timeseries rewighting with logarithmic distributions

#Arguments
- 
"""
function expectation_value_from_timeseries(P_target, P_source, list_args, list_obs)
  log_P_target(args...) = log(P_target(args...))
  log_P_source(args...) = log(P_source(args...))
  return expectation_value_from_timeseries_log(log_P_target, log_P_source, list_args, list_obs)
end

"""
Estimate distribution from a list of (measured) arguments to an (n-dimensional) probability distribution

For higher dimensional distributions (e.g. P(E,M)) list_args needs to be a list of tuples

returns:
  - Dictionary
"""
function distribution_from_timeseries_log(log_P_target, log_P_source, list_args)
  N = length(list_args)
  list_log_weight_ratio = [log_P_target(list_args[i]...)-log_P_source(list_args[i]...) for i in 1:N]
  
  log_norm = -Inf
  for i in 1:N
    log_norm  = log_sum(log_norm, list_log_weight_ratio[i])
  end
  distribution=Dict{typeof(list_args[1]),Float64}() 
  for i in 1:N
    add(distribution, list_args[i], value = exp(list_log_weight_ratio[i] - log_norm))
  end
  normalize!(distribution)
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
function expectation_value_from_histogram_log(f_args::Function, log_P_target::Function, log_P_source::Function, hist::Dict)
  log_norm = log_normalization(log_P_target, log_P_source, hist)
  println(log_norm)
  expectation_value = 0 
  for (args,H) in hist
    expectation_value += f_args(args)*H*exp(log_P_target(args...) - log_P_source(args...)-log_norm)
  end
  return expectation_value
end


function expectation_value_from_histogram_log(log_P_target::Function, log_P_source::Function, hist::Dict, hist_obs::Dict)
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
    log_norm  = log_sum(log_norm, log(H) + log_P_target(args...) - log_P_source(args...))
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
function canonical_timeseries(beta_target, beta_source, list_E, list_obs)
  N = length(list_obs)
  log_P_target(E) = -beta_target*E
  log_P_source(E) = -beta_source*E
  return timeseries_log(log_P_target,log_P_source, list_E, list_obs)
end

end

export Reweighting
