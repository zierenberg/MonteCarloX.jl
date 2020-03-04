#here tests should be written for utility functions in src/Utils.jl
using MonteCarloX
using StatsBase

function test_histogram_set_get(;verbose=false)
  pass = true

  list_ints = [i for i = 1:100]
  hist =fit(Histogram,list_ints, 1:10:101)

  target = [i for i=1:10]
  hist[1] =target[1]
  hist[12]=target[2]
  hist[23]=target[3]
  hist[34]=target[4]
  hist[45]=target[5]
  hist[56]=target[6]
  hist[67]=target[7]
  hist[78]=target[8]
  hist[89]=target[9]
  hist[100]=target[10]

  for i=1:100
    if verbose
      println("... $(hist[i]) == $(target[1+floor(Int,(i-1)/10)])")
    end
    pass &= hist[i] == target[1+floor(Int,(i-1)/10)]
  end

  return pass
end
