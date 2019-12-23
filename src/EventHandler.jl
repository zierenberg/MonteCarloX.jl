"""
# MonteCarloX.EventHandler

Module that manages event handling for a list of potential events (e.g. MC updates, or reactions) that occur with different probabilities or rates
"""
module EventHandler
using Random

"""
generate events (dt,id) from a list of rates such that their occurence corresponds with their rate
"""
function next_event_rate(list_rates::Array,rng::AbstractRNG)
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
fast implementation of next_event_rate if sum(list_rates) is known
"""
function next_event_rate(list_rates::Array,sum_rates::Float64,rng::AbstractRNG)
  dt = rand(rng,Exponential(1.0/sum_rates))
  id = 1
  theta = rand(rng)*sum_rates
  if theta < 0.5*sum_rates
    cumulated_rates = list_rates[id]
    while cumulated_rates < theta 
      id += 1
      cumulated_rates += list_rates[id]
    end
  else  
    id=length(list_rates);
    cumulated_rates_lower = sum_rates - list_rates[id]
    #cumulated_rates in this case belong to id-1
    while cumulated_rates_lower > theta
      id -= 1;
      cumulated_rates_lower -= list_rates[id]
    end
  end
  return dt,id
end

"""
generate events (id) with given probability
"""
function next_event_probability(list_probs::Array,rng::AbstractRNG)
  id = 1
  theta = rand(rng)
  if theta < 0.5
    cumulated_prob = list_probs[id]
    while cumulated_prob < theta 
      id += 1
      cumulated_prob += list_probs[id]
    end
  else  
    id = length(list_probs);
    cumulated_prob_lower = 1 - list_probs[id]
    #cumulated_rates in this case belong to id-1
    while cumulated_prob_lower > theta
      id -= 1;
      cumulated_prob_lower -= list_probs[id]
    end
  end
  return id
end


end
export EventHandler

