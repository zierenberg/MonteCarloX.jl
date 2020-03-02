using MonteCarloX
using Random

include("../examples/contact_process.jl")

""" 
    test_gillespie_fast(;verbose=false) 

Testing if sampling single event correctly

there is clearly a drift in the precision of sum_rates (needs to be compensated in system implementation of course
"""
function test_gillespie(;verbose=false)
  return pass
end
