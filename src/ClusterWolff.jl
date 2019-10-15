module ClusterWolff
using Random
using LightGraphs

#abstract type Ising

struct Ising
  dims :: Array{Int32}
  lattice :: SimpleGraph
  spins :: Array{Int32}
end

function constructIsing(dims, rng)
  lattice = LightGraphs.SimpleGraphs.grid(dims, periodic=true)
  spins = rand(rng, [-1,1], dims...)
  s = Ising(dims, lattice, spins)
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
  N = length(system.spins)

  # Get random spin and flip it
  i = rand(rng, 1:N)
  s_i = system.spins[i]
  system.spins[i] *= -1

  # depth first search in neighborhood and check for all 
  # connected neighbors j that s_j == s_i and update state with probl
  # rand(rng) < 1.0 - exp(-2*beta)
  p = 1.0 - exp(-2*beta)
  visited = BitSet([i])  
  unvisited = outneighbors(system.lattice, i)
  while !isempty(unvisited)
    j = pop!(unvisited)
    if !in(j, visited) && system.spins[j] == s_i && rand(rng) < p
      system.spins[j] *= -1
      append!(outneighbors(system.lattice, j), unvisited)
    end
    push!(visited, j)
  end
end

end


export ClusterWolff
