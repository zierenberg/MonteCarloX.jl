module ClusterWolff
using Random

#abstract type Ising

struct Ising

end


"""
Wolff single cluster update

API:
* system = of Ising type
* beta = inverse temperature
* rng = random number generator

Ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989)

consider for Potts:
https://stanford.edu/~kurinsky/ClassWork/Physics271_Final_Paper.pdf

consider BlumeCapel implementation
"""
function update(system::Ising, beta, rng)
  function replace_recursive
  end
  N = system.size
  i = rand(rng)*N
  s_i = system.state(i)
  system.state(i)*(-1) 
  #depth first search in neighborhood and check for all connected neighbors j that s_j == s_i and update state with probl
  # rand(rng)< 1.0 - exp(-2*beta)
  #for (index,value) in enumerate(system.neighborhood[i])
  #end
end

end

export ClusterWolff
