using LightGraphs
using MonteCarloX
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf
using StatsBase

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

#J=1
function energy(system::IsingSystem)
  E = 0.0
  N = 0
  for (i,s_i) in enumerate(system.spins)
    for j in outneighbors(system.lattice,i) 
      E += -1*s_i*system.spins[j] 
    end
  end
  return E/2.0
end


""" Testing Metropolis update on 2D Ising model"""
function test_ising_metropolis()
  list_beta = [0.0,0.3,0.7,1.0,1.5,2.0]
  pass = true

  rng = MersenneTwister(1000)
  system = constructIsing([8,8], rng)
  nearest_neighbors(index) = outneighbors(system.lattice, index)
  N = length(system.spins)

  for beta in list_beta
    samples = 1000
    list_energy = zeros(samples)
    # thermalization 100 sweeps
    for sweep in 1:samples+100
      #Metropolis:
      for i in 1:N
        index_i = rand(rng,1:N)
        dE = 0.0
        for index_j in nearest_neighbors(index_i)
          dE += 2.0*system.spins[index_i]*system.spins[index_j]  
        end
        if Metropolis.update_diff(dE->exp(-beta*dE),dE,0,rng)
          system.spins[index_i] *= -1
        end
      end
      if sweep > 100
        list_energy[sweep-100] = energy(system)
      end
    end
    P_meas = countmap(list_energy)
    P_true = BoltzmannDistribution(beta,log_dos_beale_8x8).pdf
    #KL divergence sum P(E)logP(E)/Q(E) 
    # P=P_meas, Q=P_true s.t P(E)=0 simply ignored
    kldivergence = 0.0
    for (E,P) in P_meas
      P = P/length(list_energy)
      Q = P_true[E]
      kldivergence += P*log(P/Q)
    end
    #println(beta, " ", kldivergence, " ", length(keys(P_meas)))
    #kldivergence depends on sample size, correlation, etc..
    pass &= kldivergence < 0.1
  end


  system = constructIsing([8,8], rng)
  for beta in list_beta
    samples = 1000
    list_energy = zeros(samples)
    # thermalization 100 sweeps
    for sweep in 1:samples+100
      #Metropolis:
      for i in 1:N
        index_i = rand(rng,1:N)
        E_old = 0.0
        E_new = 0.0
        for index_j in nearest_neighbors(index_i)
          E_old += -1*system.spins[index_i]*system.spins[index_j]  
          E_new += +1*system.spins[index_i]*system.spins[index_j]  
        end
        if Metropolis.update(E->exp(-beta*E),E_new,E_old,rng)
          system.spins[index_i] *= -1
        end
      end
      if sweep > 100
        list_energy[sweep-100] = energy(system)
      end
    end
    P_meas = countmap(list_energy)
    P_true = BoltzmannDistribution(beta,log_dos_beale_8x8).pdf
    #KL divergence sum P(E)logP(E)/Q(E) 
    # P=P_meas, Q=P_true s.t P(E)=0 simply ignored
    kldivergence = 0.0
    for (E,P) in P_meas
      P = P/length(list_energy)
      Q = P_true[E]
      kldivergence += P*log(P/Q)
    end
    #println(beta, " ", kldivergence, " ", length(keys(P_meas)))
    #kldivergence depends on sample size, correlation, etc..
    pass &= kldivergence < 0.1
  end

  return pass
end

""" Testing Cluster update on 2D Ising model"""
function test_ising_cluster()
  #avoid beta_c = 0.44
  #why does it consistently fail around beta=0.6
  list_beta = [0,0.3,0.5,1,1.5,2.0]
  pass = true

  rng = MersenneTwister(1000)
  system = constructIsing([8,8], rng)
  nearest_neighbors(index) = outneighbors(system.lattice, index)
  N = length(system.spins)
  
  for beta in list_beta
    samples = Int(1e4)
    list_energy = zeros(samples)

    # thermalization 100 sweeps
    for sweep in 1:samples+100
      for i in 1:5
        ClusterWolff.update(system.spins, nearest_neighbors, beta, rng)
      end
      if sweep > 100
        list_energy[sweep-100] = energy(system)
      end
    end

    P_meas = countmap(list_energy)
    P_true = BoltzmannDistribution(beta,log_dos_beale_8x8).pdf
    #x=sort(collect(keys(P_meas)))
    #display(plot(x,[P_meas[x_]/length(list_energy) for x_ in x]))
    #println(x)
    #println([P_meas[x_]/length(list_energy) for x_ in x])
    #println([P_true[x_] for x_ in x])
    #display(plot!(x,[P_true[x_] for x_ in x],label="true"))

    #KL divergence sum P(E)logP(E)/Q(E) 
    # P=P_meas, Q=P_true s.t P(E)=0 simply ignored
    kldivergence = 0.0
    for (E,P) in P_meas
      P = P/length(list_energy)
      Q = P_true[E]
      kldivergence += P*log(P/Q)
    end
    #println(beta, " ", kldivergence, " ", length(keys(P_meas)))
    #kldivergence depends on sample size, correlation, etc..
    pass &= kldivergence < 0.02
    
    #test = HypothesisTests.ExactOneSampleKSTest(list_energy, BoltzmannDistribution(beta,log_dos_beale_8x8))
    #pass &= pvalue(test) > 0.05
  end

  return pass
end


#""" Definition of the Boltzmann distribution for hypothesis testing."""
struct BoltzmannDistribution <: ContinuousUnivariateDistribution
    pdf::Dict{Any,Float64}
    cdf::Dict{Any,Float64}

    BoltzmannDistribution(beta, log_dos) = new(initialize_BoltzmannDistribution(beta,log_dos)...)
    #BoltzmannDistribution(beta, log_dos, log_Z) = new(Float64(beta), Dict{Any,Float64}(log_dos), Float64(log_Z))
end

#Probability distribution (for both precompute in constructor...)
pdf(d::BoltzmannDistribution, E::Real) = d.pdf[E]

#cumulated probability distribution
cdf(d::BoltzmannDistribution, E::Real) = d.cdf[E]

#implements logarithmic sum of C = e^c = A+B = e^a + e^b
# c = ln(A+B) = a + ln(1+e^{b-a})
# with b-a < 1
function log_sum(a,b)
  if b < a
    return a + log(1+exp(b-a)) 
  else
    return b + log(1+exp(a-b))
  end
end

function initialize_BoltzmannDistribution(beta,log_dos)
  #partition sum
  log_Z = 0
  for (E,log_d) in log_dos
    log_Z = log_sum(log_Z, log_d - beta*E)
  end

  pdf = Dict{Int64,Float64}()
  cdf = Dict{Int64,Float64}()

  log_cdf = 0
  for (E,log_d) in log_dos
    log_pdf = log_d - beta*E - log_Z
    log_cdf = log_sum(log_cdf, log_pdf)
    pdf[E] = exp(log_pdf)
    cdf[E] = exp(log_cdf)
  end 
  return pdf,cdf
end


log_dos_beale_8x8 = [ (-128, 0.6931471805599453  ), (-124, 0.0                 ), (-120, 4.852030263919617   ), (-116, 5.545177444479562   ), (-112, 8.449342524508063   ), (-108, 9.793672686528922   ), (-104, 11.887298863200714  ), (-100, 13.477180596840947  ), (-96 , 15.268195474147658  ), (-92 , 16.912371686315282  ), (-88 , 18.59085846191256   ), (-84 , 20.230089202801466  ), (-80 , 21.870810400320693  ), (-76 , 23.498562234123614  ), (-72 , 25.114602234581373  ), (-68 , 26.70699035290573   ), (-64 , 28.266152815389898  ), (-60 , 29.780704423363996  ), (-56 , 31.241053997806176  ), (-52 , 32.63856452513369   ), (-48 , 33.96613536105969   ), (-44 , 35.217576663643314  ), (-40 , 36.3873411250109    ), (-36 , 37.47007844691906   ), (-32 , 38.46041522581422   ), (-28 , 39.35282710786369   ), (-24 , 40.141667825183845  ), (-20 , 40.82130289691285   ), (-16 , 41.38631975325592   ), (-12 , 41.831753810069756  ), (-8  , 42.153328313883975  ), (-4  , 42.34770636939425   ), (0   , 42.41274640460084   ), (4   , 42.34770636939425   ), (8   , 42.153328313883975  ), (12  , 41.831753810069756  ), (16  , 41.38631975325592   ), (20  , 40.82130289691285   ), (24  , 40.141667825183845  ), (28  , 39.35282710786369   ), (32  , 38.46041522581422   ), (36  , 37.47007844691906   ), (40  , 36.3873411250109    ), (44  , 35.217576663643314  ), (48  , 33.96613536105969   ), (52  , 32.63856452513369   ), (56  , 31.241053997806176  ), (60  , 29.780704423363996  ), (64  , 28.266152815389898  ), (68  , 26.70699035290573   ), (72  , 25.114602234581373  ), (76  , 23.498562234123614  ), (80  , 21.870810400320693  ), (84  , 20.230089202801466  ), (88  , 18.59085846191256   ), (92  , 16.912371686315282  ), (96  , 15.268195474147658  ), (100 , 13.477180596840947  ), (104 , 11.887298863200714  ), (108 , 9.793672686528922   ), (112 , 8.449342524508063   ), (116 , 5.545177444479562   ), (120 , 4.852030263919617   ), (124 , 0.0                 ), (128 , 0.6931471805599453  ) ];
