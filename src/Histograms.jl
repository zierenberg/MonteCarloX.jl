#TODO: Think about using StatsBase.Histograms ... -> not really my interface because a priori one has to specify the histogram ..
# -> add with push!(hist, x) and this might actually be decently fast
# hmm StatsBase is of course faster -> replace then ...

"""
    Histogram(list_x::Vector{Tx}, dx::Tx; increment::Tv=1)::Dict{Tx,Tv} where {Tx<:Real, Tv<:Real}

One-dimensional histogram H(x) from a list of `x` values, sorted into bins of
size `dx` and adding `increment` to respecive bin when value of `x` occurs in
list.

If `dx = "none"` then `x` are assumed to be discrete and a new entry for every
new unique `x` is created. This can cause heavy memory usage for continous
variables, where typically no two values coincide perfectly.

TODO: test performance of this when we drop the type stability thing
TODO: multuple case could be done with NTuple{N,Tx} where {N,Tx}
TODO: think how StatsBase.Histogram can be used for this but so far the access of the elements is not optimal
TODO: this has to be discussed when implementing muca and such, e.g., is it faster to calculate element in rnage or access dictionary?
      !!! can a dictionary access intermediates?
      !!! this is also not optimal for us, because add only adds to existing keys
"""
function Histogram(list_x::Vector{Tx}; dx::Union{Tx,String}="none", increment::Tv=1)::Dict{Tx,Tv} where {Tx<:Real, Tv<:Real}
  hist = Dict{Tx,Tv}()
  if dx == "none"
    for x in list_x
      add!(hist, x, increment=increment)
    end
  else
    for x in list_x
      #this clamps x_bin to the lower bound of x/dx
      x_bin = floor(x/dx)*dx
      #x_bin = floor.(x./dx).*dx for multidimensional case
      add!(hist, x_bin, increment=increment)
    end
  end
  return hist
end

"""
    Distribution(list_x::Vector{Tx}, dx::Union{Tx,String})::Dict{Tx,Float64} where  {Tx<:Real}

One-dimensional distribution `P(x)` with bin width `dx` such that ``\\sum_x P(x)dx = 1``.
"""
function Distribution(list_x::Vector{Tx}, dx::Union{Tx,String})::Dict{Tx,Float64} where  {Tx<:Real}
  increment = 1.0/length(list_x)/dx
  return Histogram(list_x, dx=dx, increment=increment)
end

"""
    add!(hist::Dict, args; increment=1)

Add `increment` to the value of `hist[args]`

The default increment is 1 as would be expected for a histogram, while an explicit choice for `increment` can be set to directly generate a normalized distribtion.

TODO: This is not very type stable ... does it need to?
"""
function add!(hist::Dict, args; increment=1)
  if args in keys(hist)
    hist[args] += increment
  else
    hist[args]  = increment
  end
end


"""
    normalize!(dist::Dict; dx::T = 1.0) where {T <: Real}    

Normalize a distribution `dict` such that ``\\sum_x dict[x] dx = 1`` 

The default `dx` is 1.0 assuming discrete 

TODO: This is not very type stable ... does it need to?
"""
function normalize!(dist::Dict; dx=1.0)
  norm = 0.0
  for (args, P) in dist
    norm += P
  end

  norm *= dx

  for (args, P) in dist
    dist[args] = P/norm
  end
end
