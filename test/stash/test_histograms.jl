using MonteCarloX

function test_histogram(;verbose=false)
  pass = true

  list_ints = [i for i = 1:100]
  hist = MonteCarloX.Histogram(list_ints , dx="none")

  add!(hist,1)
  add!(hist,2)

  sum_values = 0.0
  for (x, H) in hist
    sum_values += H
  end

  pass &= sum_values == 102
  pass &= hist[1] == 2
  pass &= hist[2] == 2
  pass &= hist[3] == 1
  return pass

end

function test_distribution(;verbose=false)
end
