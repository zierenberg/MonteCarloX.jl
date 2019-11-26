using LightGraphs
using Distributions
using MonteCarloX
using Random

function run(N::Int,m::Float64,h::Float64,p::Float64,T::Float64,T_therm::Float64,seed::Int; flag_fast = true)
  rng = MersenneTwister(seed);
  rng2 = MersenneTwister(seed);

  # directed ER graph
  neuron = zeros(N);
  graph = LightGraphs.SimpleGraphs.erdos_renyi(N, p, is_directed=true, seed=seed)
  #graph.fadjlist gives outgoing nodes
  #inneighbors(graph,v) -> works
  
  # number of active neighbors for fast update of local lambdi_i
  neuron_active_neighbors = zeros(Int8,N);
  # rate array
  rate = ones(Float64,N).*h;
  R = sum(rate)

  #dimensionless rates
  mu = 1
  lambda = mu*m

  #loop over time
  T_total = T + T_therm;
  time = 0
  dt   = 0
  Nact = 0
  dtime      = 0;
  dtime_sum  = 0;
  dtime_step = 10;
  N_meas     = 0;
  avg_activity  = 0;
  avg_activity2 = 0;
  while time < T_total
    # find next event (dt and time) to be updated
    if flag_fast
      dt,n = Gillespie.next_event_fast(rate,R,rng)
    else
      dt,n = Gillespie.next_event(rate,rng)
    end
    # measure observables in discrete time 
    # use values from last step because they were valid inbetween
    dtime += dt;
    while dtime > dtime_step
      dtime_sum += dtime_step;
      dtime     -= dtime_step;
      if time > T_therm  
        avg_activity  += Nact;
        avg_activity2 += Nact*Nact;
        N_meas        += 1;
      end
    end
    # evolve time 
    time += dt;
    # update neuron n, rate list, sum of rates R,
    rate_n_old = rate[n];
    dstate = 0;
    if neuron[n]==0  
      neuron[n] = 1;
      rate[n] = mu;
      dstate  = 1;
    else  
      neuron[n] = 0;
      rate[n] = h + lambda*neuron_active_neighbors[n]/length(inneighbors(graph,n)) 
      dstate  = -1;
    end
    #update number of active nodes
    Nact += dstate;
    #update rate list
    R += (rate[n] - rate_n_old);
    #update neighboring neurons
    for nn in outneighbors(graph,n)
      neuron_active_neighbors[nn] += dstate;
      if neuron[nn]==0  
        rate_nn_old = rate[nn];
        rate[nn] = h + lambda*neuron_active_neighbors[nn]/length(inneighbors(graph,n)) 
        R += (rate[nn] - rate_nn_old); 
      end
    end
  end

  return avg_activity/N_meas, avg_activity2/N_meas, N_meas
end

##############################run
#avg_activity, avg_activity2, T = run(Int(1e3),1.0,1e-3,1e-2,1e6,1e3,rng)
#println(avg_activity, " ", avg_activity2, " ", T)

#err = sqrt((avg_activity2-avg_activity*avg_activity)/(T-1))
#println(avg_activity, " ", err, " ", T)


