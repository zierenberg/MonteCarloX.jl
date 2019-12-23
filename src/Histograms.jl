module Histograms
"""
create a histogram from a list of values

can this be formulated for list_args, dargs, args_ref
"""
function histogram(list_x, dx, x_ref; increment=1)
  hist = Dict()
  for x in list_x
    x_bin = x_ref + floor((x-x_ref)/dx)*dx
    add(hist, x_bin, increment=increment)
  end
  return hist
end

function histogram(list_x; increment=1)
  hist = Dict()
  for x in list_x
    add(hist, x, increment=increment)
  end
  return hist
end

"""
create a distribution (sum_args dist[args] = 1) from a list of values
"""
function distribution(list_x, dx, x_ref)
  increment = 1/length(list_x)/dx
  return histogram(list_x,dx,x_ref,increment=increment)
end

function distribution(list_x)
  increment = 1.0/length(list_x)/dx
  return histogram(list_x,increment=increment)
end

"""
add a value to a histogram dictionary

per default this adds 1 but can be used for distributions as well when value is set explicitly 
"""
function add(hist::Dict, args; increment=1)
  if args in keys(hist)
    hist[args] += increment
  else
    hist[args]  = increment
  end
end

end
export Histograms
