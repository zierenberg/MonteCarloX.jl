using StatsBase
import StatsBase.kldivergence
using LinearAlgebra

"""
    kldivergence(P::Histogram, Q::Function)

Kullback-Leibler divergence between an empirical distribution (measured) and a reference distribution (analytic)

So far this is defined only for 1-dimensional distributions of type StatsBase.Histogram
"""
function kldivergence(P::Histogram, Q::Function)
    ##KL divergence sum P(args)logP(args)/Q(args) 
    ## P=P_meas, Q=P_true s.t P(args)=0 simply ignored
    kld = 0.0
    for (i,x) in enumerate(P.edges[1])
      if i <= length(P.weights)
        @inbounds p = P.weights[i]
        if p > 0
          q = Q(x)
          kld += p*log(p/q)
        end
      end
    end
    return kld
end

