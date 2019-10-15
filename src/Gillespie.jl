module Gillespie
using Random

"""
Gillespie update

API - input
* list of rates
* rng

API - output
* returns element that is to be updated

"""
function update(list_rates,R,rng)
  theta = rand(rng)
  i=1
  if theta < 0.5
    i=1
    sum_rate = list_rates[i]/R
    while sum_rate < theta 
      i+=1
      sum_rate += list_rates[i]/R
    end
  else  
    i=length(list_rates);
    sum_rate = 1.0 - list_rates[i]/R;
    while sum_rate > theta
      i -= 1;
      sum_rate -= list_rates[i]/R;
    end
  end
  return i
end

end

export Gillespie
