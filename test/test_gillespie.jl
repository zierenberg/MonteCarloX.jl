using MonteCarloX
using Random

include("../examples/contact_process.jl")

""" 
Testing if sampling single event correctly

there is clearly a drift in the precision of sum_rates
"""
function test_gillespie_fast(;verbose=false)
  pass = true
  fst_a, fst_a2 = run(Int(1e3),1.0,1e-3,1e-2,1e3,1e1,1000, flag_fast=true)
  std_a, std_a2 = run(Int(1e3),1.0,1e-3,1e-2,1e3,1e1,1000, flag_fast=false)
  
  if verbose
    println("result average activity fast ($(fst_a)) and slow ($(std_a))")
  end

  pass &= abs(fst_a-std_a) < 1e-8
  pass &= abs(fst_a2-std_a2) < 1e-8
  return pass

end
