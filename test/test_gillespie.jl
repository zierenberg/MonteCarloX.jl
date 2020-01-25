using MonteCarloX
using Random

include("../examples/contact_process.jl")

""" 
    test_gillespie_fast(;verbose=false) 

Testing if sampling single event correctly

there is clearly a drift in the precision of sum_rates (needs to be compensated in system implementation of course
"""
function test_gillespie_fast(;verbose=false)
  pass = true
  fst_a, fst_a2 = run(Int(1e2),1.0,1e-3,1e-2,1e5,1e1,1000, flag_fast=true)
  std_a, std_a2 = run(Int(1e2),1.0,1e-3,1e-2,1e5,1e1,1000, flag_fast=false)
  
  if verbose
    println("result average activity fast ($(fst_a)) and slow ($(std_a))")
  end

  pass &= abs(fst_a-std_a) < 1e-8
  pass &= abs(fst_a2-std_a2) < 1e-8
  return pass

end
