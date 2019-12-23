using LightGraphs
using MonteCarloX
using Random
using HypothesisTests
using Distributions
import Distributions.pdf
import Distributions.cdf
using StatsBase
import StatsBase.kldivergence


#TODO: use this also for PT and adaptive overlap stories etc

""" Testing metropolis on unimodaL probability distribution"""
function test_unimodal_metropolis()
  P_unimodal(x,x0,var) = exp(-(x-x0)^2/2/var)/sqrt(2*pi*var)
  var = 1
  dx = 0.1
  list_x0 = [0,1,2,3]
  pass = true

  rng = MersenneTwister(1000)

  for x0 in list_x0
    P_sampling(x) = P_unimodal(x,x0,var)
    samples = Int(1e6)
    list_x = zeros(samples)
    x = x0
    # thermalization 100 sweeps
    for sweep in 1:samples+100
      #Metropolis:
      x_new = step(x,dx,rng)
      if Metropolis.accept(P_sampling, x_new, x,rng)
        x = x_new
      end
      if sweep > 100
        list_x[sweep-100] = x 
      end
    end
    # sort into binned histogram
    # bin size/ 2*sigma ~  #elements per bin (~100)/ samples*0.6/
    bin_x =  1e4*sqrt(var)*2/0.6/samples
    P_meas = Histograms.distribution(list_x, bin_x, x0)
    kld = abs(kldivergence(P_meas, P_sampling))
    println("result kldivergence P_meas to P_true = $(kld)")
    pass &= kld < 0.1
    ###########################################################################
    est_x0 = sum(list_x)/length(list_x)
    diff_est_x0 = abs(est_x0 - x0)
    println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
    pass &= diff_est_x0 < 0.1
  end
  return pass
end

#TODO: test speed difference
""" Testing sweep on unimodaL probability distribution"""
function test_unimodal_sweep()
  P_unimodal(x,x0,var) = exp(-(x-x0)^2/2/var)/sqrt(2*pi*var)
  var = 1
  dx = 0.1
  list_x0 = [0,1,2,3]
  pass = true


  rng = MersenneTwister(1000)


  for x0 in list_x0
    P_sampling(x) = P_unimodal(x,x0,var)
    samples = Int(1e5)
    list_x = zeros(samples)

    system = System(x0)
    stats  = Stats(0,0)
    list_updates = [()->update(P_sampling, system, 0.1, rng, stats),
                    ()->update(P_sampling, system, 0.2, rng)       ,
                    ()->update(P_sampling, system, 0.1, rng)        ]
    list_probabilities = [0.2,0.4,0.4]

    #perfrom 100 updates as thermalization
    MonteCarloX.sweep(list_updates, list_probabilities, rng, number_updates=100)

    for updates in 1:samples
      MonteCarloX.sweep(list_updates, list_probabilities, rng, number_updates=10)
      list_x[updates] = system.x 
    end
    # sort into binned histogram
    # bin size/ 2*sigma ~  #elements per bin (~100)/ samples*0.6/
    bin_x =  1e4*sqrt(var)*2/0.6/samples
    P_meas = Histograms.distribution(list_x, bin_x, x0)
    kld = abs(kldivergence(P_meas, P_sampling))
    println("result kldivergence P_meas to P_true = $(kld)")
    pass &= kld < 0.1
    ###########################################################################
    est_x0 = sum(list_x)/length(list_x)
    diff_est_x0 = abs(est_x0 - x0)
    println("result estimate x0 = $(est_x0) vs x0 = $(x0) [diff = $(diff_est_x0)]")
    pass &= diff_est_x0 < 0.1
  end
  return pass
end


""" Testing reweighting on unimodaL probability distribution"""
function test_unimodal_reweighting()

  return pass
end


###############################################################################
###############################################################################
###############################################################################
 
mutable struct System
  x :: Float64
end

mutable struct Stats
  accept::Float64
  reject::Float64
end


function update(P_sampling, system::System, dx, rng::AbstractRNG)
  x_new = step(system.x,dx,rng)
  if Metropolis.accept(P_sampling, x_new, system.x,rng)
    system.x = x_new
  end
end

#TODO: test second version with statistics could this be implemented in update w/o speed loss?
function update(P_sampling, system::System, dx, rng::AbstractRNG, stats::Stats)
  x_new = step(system.x,dx,rng)
  if Metropolis.accept(P_sampling, x_new, system.x,rng)
    system.x = x_new
    stats.accept += 1
  else 
    stats.reject += 1
  end
end

function step(x,dx,rng)
  return x -dx + 2*dx*rand(rng)
end



"""
Kullback-Leibler divergence between two distributions

distributions are here considered to be dictionaries

valid for n-dimensional dictionaries (args=Tuple or larger?)
"""
function kldivergence(P::Dict,Q::Function)
    ##KL divergence sum P(args)logP(args)/Q(args) 
    ## P=P_meas, Q=P_true s.t P(args)=0 simply ignored
    #
    kld = 0.0
    for (args,p) in P
      q = Q(args)
      kld += p*log(p/q)
    end
    return kld
end

