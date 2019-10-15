module Metropolis
using Random

"Metropolis update comparing new energy (E_new) with old energy (E_old) for a given inverse temperature beta using parsed random number generator that should be Float64 type"
function update(E_new,E_old,beta,rng)
  dE = E_new - E_old
  if dE<0
    return true
  elseif rand(rng)<exp(-beta*dE)
    return true
  else
    return false
  end
end

end

export Metropolis

