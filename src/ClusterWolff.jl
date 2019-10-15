module ClusterWolff
using Random
using LightGraphs

struct IsingSystem
  dims :: Array{Int32}
  lattice :: SimpleGraph
  spins :: Array{Int32}
end

function constructIsing(dims, rng)
  lattice = LightGraphs.SimpleGraphs.grid(dims, periodic=true)
  spins = rand(rng, [-1,1], dims...)
  s = IsingSystem(dims, lattice, spins)
  return s
end

function magnetization(system::IsingSystem)
  return abs(sum(system.spins))
end

function testRun(beta)

  function m(N, beta, nSteps)
    rng = MersenneTwister(1000)
    system = constructIsing([N,N], rng)
    
    meanMagnetization = 0.0
    for i = 1:nSteps
      update(system, beta, rng)
      #println(system.spins)
      meanMagnetization += magnetization(system)
    end
    meanMagnetization /= nSteps

    return meanMagnetization
  end

  return [m(10, beta, 50000) for beta in 0.0:0.05:2.0]
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
function update(system::IsingSystem, beta, rng)
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
  unvisited = copy(outneighbors(system.lattice, i))
  while !isempty(unvisited)
    j = pop!(unvisited)
    if !in(j, visited) && system.spins[j] == s_i && rand(rng) < p
      system.spins[j] *= -1
      append!(unvisited, outneighbors(system.lattice, j))
    end
    push!(visited, j)
  end
end

end


export ClusterWolff
