
"""
add a value to a histogram dictionary

per default this adds 1 but can be used for distributions as well when value is set explicitly 
"""
function add(hist::Dict, args; value=1)
  if args in keys(hist)
    hist[args] += value
  else
    hist[args]  = value
  end
end
