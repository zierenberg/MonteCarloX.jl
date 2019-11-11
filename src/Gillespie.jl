module Gillespie
using Random
using Distributions

"""
Gillespie update
as wrapper around inhomogeneous Poisson process? 

# Arguments
- `list_rates`: list of constant rates until next event happens 
- `rng`: random number generator 

# Output
* returns next event (dt, id)

"""
function next_event(list_rates,rng)
  #sum_rates=sum(list_rates)
  #return next_event_fast(list_rates, sum_rates, rng)
  cumulated_rates = cumsum(list_rates)
  sum_rates = cumulated_rates[end]
  dt = rand(rng,Exponential(1.0/sum_rates))
  theta = rand(rng)*sum_rates
  #catch lower-bound case that cannot be reached by binary search
  if theta < cumulated_rates[1] 
    id = 1
  else 
    #binary search
    index_l = 1
    index_r = length(cumulated_rates)
    while index_l < index_r-1
      index_m = floor(Int,(index_l+index_r)/2)
      if cumulated_rates[index_m] < theta
        index_l = index_m
      else
        index_r = index_m
      end
    end
    id = index_r
  end
  return dt, id
end



"""
Gillespie update without require loop over full rate array

# Arguments
- `list_rates`: list of constant rates until next event happens 
- `sum_rates`: sum over list of rates maintained outside of function
- `rng`: random number generator 

# Output
* returns next event (dt, id)

"""
function next_event_fast(list_rates,sum_rates,rng)
  dt = rand(rng,Exponential(1.0/sum_rates))
  id = 1
  theta = rand(rng)*sum_rates
  if theta < 0.5*sum_rates
    cumulated_rates = list_rates[id]
    while cumulated_rates < theta 
      id += 1
      cumulated_rates += list_rates[id]
    end
    #if (cumulated_rates < theta) || (cumulated_rates - list_rates[id] > theta)
    #  println("ERROR: fast forward")
    #end
  else  
    id=length(list_rates);
    cumulated_rates_lower = sum_rates - list_rates[id]
    #cumulated_rates in this case belong to id-1
    while cumulated_rates_lower > theta
      id -= 1;
      cumulated_rates_lower -= list_rates[id]
    end
    #if (cumulated_rates_lower > theta) || (cumulated_rates_lower + list_rates[id] < theta)
    #  println("ERROR: fast backwards")
    #end
  end
  return dt,id
end

end

export Gillespie
