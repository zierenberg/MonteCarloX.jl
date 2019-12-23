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

TODO: add sth lik (... ; stats=None) where things like acceptance rates etc can be cumulated in stats
TODO: rename update check/accept?
"""
function accept(P,args_new,args_old,rng::AbstractRNG)
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
function accept_diff(P,args_new,args_old,rng)
  args_diff = args_new .- args_old
  if rand(rng) < P(args_diff...)
    return true
  else
    return false
  end
end

##TODO: Think about making everything in log?
#function update(P,system,update,rng; stats=empty)
#  args_new, args_old, update_diff = update(system, rng)
#  if Metropolis.accept(P,args_new, args_old, rng)
#    #accept
#    stats.accept += 1
#  else
#    #reject
#    stats.reject += 1
#    update_undo(system, update_diff)
#  end
#end
#
#function sweep(P_sampling, list_updates, system, rng, number_updates=10)
#  for i in 1:number_updates
#    id = next_event_probability(list_probabilities)
#  end
#end


end
export Metropolis

