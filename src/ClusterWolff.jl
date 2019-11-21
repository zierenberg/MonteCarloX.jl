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

#J=1
function energy(system::IsingSystem)
  E = 0.0
  N=0
  for (i,s_i) in enumerate(system.spins)
    for j in outneighbors(system.lattice,i) 
      E += -1*s_i*system.spins[j] 
    end
  end
  return E/2
end

function testRun()
  log_dos_beale_8x8=[
                    (-128, 0.6931471805599453  )
                    (-120, 4.852030263919617   )
                    (-116, 5.545177444479562   )
                    (-112, 8.449342524508063   )
                    (-108, 9.793672686528922   )
                    (-104, 11.887298863200714  )
                    (-100, 13.477180596840947  )
                    (-96 , 15.268195474147658  )
                    (-92 , 16.912371686315282  )
                    (-88 , 18.59085846191256   )
                    (-84 , 20.230089202801466  )
                    (-80 , 21.870810400320693  )
                    (-76 , 23.498562234123614  )
                    (-72 , 25.114602234581373  )
                    (-68 , 26.70699035290573   )
                    (-64 , 28.266152815389898  )
                    (-60 , 29.780704423363996  )
                    (-56 , 31.241053997806176  )
                    (-52 , 32.63856452513369   )
                    (-48 , 33.96613536105969   )
                    (-44 , 35.217576663643314  )
                    (-40 , 36.3873411250109    )
                    (-36 , 37.47007844691906   )
                    (-32 , 38.46041522581422   )
                    (-28 , 39.35282710786369   )
                    (-24 , 40.141667825183845  )
                    (-20 , 40.82130289691285   )
                    (-16 , 41.38631975325592   )
                    (-12 , 41.831753810069756  )
                    (-8  , 42.153328313883975  )
                    (-4  , 42.34770636939425   )
                    (0   , 42.41274640460084   )
                    (4   , 42.34770636939425   )
                    (8   , 42.153328313883975  )
                    (12  , 41.831753810069756  )
                    (16  , 41.38631975325592   )
                    (20  , 40.82130289691285   )
                    (24  , 40.141667825183845  )
                    (28  , 39.35282710786369   )
                    (32  , 38.46041522581422   )
                    (36  , 37.47007844691906   )
                    (40  , 36.3873411250109    )
                    (44  , 35.217576663643314  )
                    (48  , 33.96613536105969   )
                    (52  , 32.63856452513369   )
                    (56  , 31.241053997806176  )
                    (60  , 29.780704423363996  )
                    (64  , 28.266152815389898  )
                    (68  , 26.70699035290573   )
                    (72  , 25.114602234581373  )
                    (76  , 23.498562234123614  )
                    (80  , 21.870810400320693  )
                    (84  , 20.230089202801466  )
                    (88  , 18.59085846191256   )
                    (92  , 16.912371686315282  )
                    (96  , 15.268195474147658  )
                    (100 , 13.477180596840947  )
                    (104 , 11.887298863200714  )
                    (108 , 9.793672686528922   )
                    (112 , 8.449342524508063   )
                    (116 , 5.545177444479562   )
                    (120 , 4.852030263919617   )
                    (128 , 0.6931471805599453  )
                   ]
  function e_ana_8(beta)
    mean_energy = 0.0
    norm = 0.0
    for (E,log_dos) in log_dos_beale_8x8
      mean_energy += E*exp(log_dos-beta*E)
      norm += exp(log_dos-beta*E)
    end
    return mean_energy/norm
  end

  function m_sim_8(beta, nSteps)
    rng = MersenneTwister(1000)
    system = constructIsing([N,N], rng)
    
    meanMagnetization = 0.0
    for i = 1:nSteps
      update(system, beta, rng)
      meanMagnetization += magnetization(system)
    end
    meanMagnetization /= nSteps

    return meanMagnetization
  end
  
  function e_sim_8(beta, nSteps)
    rng = MersenneTwister(1000)
    system = constructIsing([8,8], rng)
    
    meanEnergy = 0.0
    for i = 1:nSteps
      #sweeep
      for j = 1:length(system.lattice)
        update(system, beta, rng)
      end
      #not efficient ... try to access dE 
      meanEnergy += energy(system)
    end
    meanEnergy /= nSteps

    return meanEnergy
  end

  array_e_sim = [e_sim_8(beta, 10000) for beta in 0.0:0.05:2.0]
  array_e_ana = [e_ana_8(beta) for beta in 0.0:0.05:2.0]
  println(array_e_ana)
  println(array_e_sim)
  return (array_e_sim .- array_e_ana)./array_e_ana
end

"""
Wolff single cluster update

#Arguments:
* `spins` : array of spin values
* `nearest_neighbors` : function that returns a list of nearest neighbors to index i
* `beta` : inverse temperature
* `rng`  : random number generator

Ref: U. Wolff, Phys. Rev. Lett. 62, 361 (1989)

consider for Potts:
https://stanford.edu/~kurinsky/ClassWork/Physics271_Final_Paper.pdf

consider BlumeCapel implementation

todo: return dE?

Problem: this is inconsistent API....still?
"""
function update(spins::Array{Int32}, nearest_neighbors::Any, beta, rng)
  N = length(spins)

  # Get random spin and flip it
  i = rand(rng, 1:N)
  s_i = spins[i]
  spins[i] *= -1

  # depth first search in neighborhood and check for all 
  # connected neighbors j that s_j == s_i and update state with probl
  # rand(rng) < 1.0 - exp(-2*beta)
  p = 1.0 - exp(-2*beta)
  visited = BitSet([i])  
  unvisited = copy(nearest_neighbors(i))
  while !isempty(unvisited)
    j = pop!(unvisited)
    if !in(j, visited) && spins[j] == s_i && rand(rng) < p
      spins[j] *= -1
      append!(unvisited, nearest_neighbors(j))
    end
    push!(visited, j)
  end
end

end


export ClusterWolff
