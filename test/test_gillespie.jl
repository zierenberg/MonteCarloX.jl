using MonteCarloX
using Random

include("../examples/contact_process.jl")

""" Testing if sampling single event correctly"""
function test_gillespie_fast()
  pass = true
  fst_a, fst_a2 = run(Int(1e3),1.0,1e-3,1e-2,1e5,1e1,1000, flag_fast=true)
  std_a, std_a2 = run(Int(1e3),1.0,1e-3,1e-2,1e5,1e1,1000, flag_fast=false)

  pass &= abs(fst_a-std_a) < 1e-8
  pass &= abs(fst_a2-std_a2) < 1e-8
  return pass

end
