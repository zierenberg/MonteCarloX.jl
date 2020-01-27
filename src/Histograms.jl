"""
    MonteCarloX.Histograms

Module represents histograms/distributions as dictionaries and handles addition
"""
module Histograms

#TODO: make all this type stable (generalize to tuples? not sure how to properly but do this maybe in second realization?)
#TODO: why do I need this x_ref version?
"""
create a histogram from a list of values

can this be formulated for list_args, dargs, args_ref
"""
function histogram(list_x::Vector{Tx}, dx::Tx, x_ref::Tx; increment::Tv=1)::Dict{Tx,Tv} where {Tx<:Real, Tv<:Real}
  hist = Dict{Tx,Tv}()
  for x in list_x
    #this clamps x_bin to the lower bound of x0,x0+dx,...
    x_bin = x_ref + floor((x-x_ref)/dx)*dx
    add(hist, x_bin, increment=increment)
  end
  return hist
end

function histogram(list_x::Vector{Tx}; increment::Tv=1)::Dict{Tx,Tv} where {Tx<:Real, Tv<:Real}
  hist = Dict{Tx,Tv}()
  for x in list_x
    add(hist, x, increment=increment)
  end
  return hist
end

"""
create a distribution (sum_args dist[args] = 1) from a list of values
"""
function distribution(list_x::Vector{Tx}, dx::Tx, x_ref::Tx)::Dict{Tx,Float64} where  {Tx<:Real}
  increment = 1.0/length(list_x)/dx
  return histogram(list_x,dx,x_ref,increment=increment)
end

function distribution(list_x::Vector{T})::Dict{T,Float64} where T<:Real
  increment = 1.0/length(list_x)
  return histogram(list_x,increment=increment)
end

"""
add a value to a histogram dictionary

per default this adds 1 but can be used for distributions as well when value is set explicitly 
"""
#TODO: make this typestable despite multidimensional args?
function add(hist::Dict, args; increment=1)
  if args in keys(hist)
    hist[args] += increment
  else
    hist[args]  = increment
  end
end

end
export Histograms
