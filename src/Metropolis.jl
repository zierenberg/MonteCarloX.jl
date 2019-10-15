module Metropolis
using Random

#"Metropolis update comparing new energy (E_new) with old energy (E_old) for a given inverse temperature beta using parsed random number generator that should be Float64 type"
#function update(E_new,E_old,beta,rng)
#  dE = E_new - E_old
#  if dE<0
#    return true
#  elseif rand(rng)<exp(-beta*dE)
#    return true
#  else
#    return false
#  end
#end

"""
Metropolis update probabilities of new and old states taking multiple arguments defined by model

this could also include the selection bias (Hastings)
"""
function update(args_new,args_old,P,rng)
  ratio = P(args_new...)/P(args_old...)
  if ratio > 1
    return true
  elseif rand(rng)<ratio
    return true
  else
    return false
  end
end

"""
Metropolis update probabilities of new and old states taking multiple arguments defined by model

PLEASE BE CAREFUL AND KNOW WHAT YOU DO
"""
function update_diff(args_new,args_old,P,rng)
  args_diff = args_new .- args_old
  if rand(rng) < P(args_diff...)
    return true
  else
    return false
  end
end

end

export Metropolis

