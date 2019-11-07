module InhomogeneousPoissonProcess
using Random

"""
Generate a new event from an inhomogeneous poisson process with rate Lambda(t).
Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  2) 
see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf

# Arguments
- `rate`: rate(dt) has to be defined outside (e..g t-> rate(t-t0,args) 
- `rate_max`: maximal rate in near future (has to be evaluated externally)
- `rng`: random number generator 

API - output
* returns the next event time

"""
function next_event_time(rate, max_rate::Float64, rng::AbstractRNG)
  dt = 0
  while true
    # generate next event from bounding homogeneous Poisson process with rate_max
    dt += rand(rng,Exponential(max_rate))
    # accept next event with probability rate(t)/rate_max [Thinning algorithm]
    if rand(rng) < rate(dt)/max_rate
      return dt
    end
  end
end

#TODO
function next_event_time_for_piece_wise_decreasing_rate(rate, rng::AbstractRNG)

end

"""
event = (time,id)
TODO: BETTER NAME AND CHECK SPEED
"""
function next_event_for_collection(rates, max_rate::Float64, rng::AbstractRNG)
  rate(t)=sum(map(f->f(t),rates))
  next_time = next_event_time(rate, max_rate, rng)
  
  theta = rand(rng)
  next_index = 1
  cumulated_rates = cumsum(rates)
  sum_rate = cumulated_rates[end]
  if theta < 0.5
    while cumulated_rates[next_index]/sum_rate < theta 
      next_index +=1
    end
  else  
    next_index = length(list_rates);
    while cumulated_rates[next_index]/sum_rate > theta
      next_index -= 1;
    end
  end
end

export InhomogeneousPoissonProcess
