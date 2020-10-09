using LightGraphs
using Distributions
using MonteCarloX
using Random
using StatsBase

# this runs 5-7s for run(1000,0.1,1.0,1e-3,1e5,1/1e-3,1000) similar to my C code (5s)
# -> (23.4668, 888.0654, 10000)
function run(N::Int, p::Float64, m::Float64, h::Float64, T::Float64, T_therm::Float64, seed::Int; flag_fast = true)::Tuple{Float64,Float64,Int}
    rng = MersenneTwister(seed);

    # dimensionless rates
    mu     = 1.0
    lambda = mu * m

    system = construct_system(N, p, mu, lambda, h, seed, initial = "empty")

    T_total = T + T_therm;
    time = dt = dtime = 0.0;
    dtime_step = 10.0;
    N_active = N_meas = 0;
    avg_activity = avg_activity2 = 0.0;

    events_since_last_sum = 0
    while time < T_total
    # find next event (dt and time) to be updated
        #dt, n = next(KineticMonteCarlo(), system.rates, rng)
        dt, n = next(KineticMonteCarlo(), system, rng) # this has many more allocations than the other???
        events_since_last_sum += 1 
        if events_since_last_sum > 1000 
            system.sum_rates = sum(system.rates)
            events_since_last_sum = 0
        end

        # measure observables in discrete time 
        # use values from last step because they were valid inbetween
        dtime += dt;
        while dtime > dtime_step
            dtime     -= dtime_step
            if time > T_therm  
                avg_activity  += N_active
                avg_activity2 += N_active * N_active
                N_meas        += 1
            end
        end
    
        # only after measurement evolve real time
        time += dt;

        N_active += update(system, n)
    end

    return avg_activity / N_meas, avg_activity2 / N_meas, N_meas
end

# crucial here is type-stability achieved with F1 and F2!!!
mutable struct ContactProcess{F1,F2}
    network::SimpleDiGraph{Int64}
    incoming_neighbors::F1
    outgoing_neighbors::F2
    neurons::Vector{Int8}
    active_incoming_neighbors::Vector{Int8}
    rates::Vector{Float64}  # static list of rates
    sum_rates::Float64
    mu::Float64
    lambda::Float64
    h::Float64
end

# here one could select initial condition empty, full, equilibrium?
function construct_system(N, p, mu, lambda, h, seed; initial = "empty")::ContactProcess
    # directed ER graph
    network = LightGraphs.SimpleGraphs.erdos_renyi(N, p, is_directed = true, seed = seed)
    incoming_neighbors(n) = inneighbors(network, n)
    outgoing_neighbors(n) = outneighbors(network, n)
  
    # initial conditions 
    neurons = []
    active_incoming_neighbors = []
    if initial == "empty"
        neurons = zeros(Int8, N)
        active_incoming_neighbors = zeros(Int8, N);
        rates = ones(Float64, N) .* h;
    end
    if initial == "full"
        neurons = ones(Int8, N);
        active_incoming_neighbors = ones(Int8, N);
        rates = ones(Float64, N) .* mu;
    end
    sum_rates = sum(rates)

    return ContactProcess(network, incoming_neighbors, outgoing_neighbors, neurons, active_incoming_neighbors, rates, sum_rates, mu, lambda, h)
end


function Base.sum(system::ContactProcess)
    return system.sum_rates
end

function Base.getindex(system::ContactProcess, index)
    return system.rates[index]
end

function Base.length(system::ContactProcess)
    return length(system.rates)
end

# system has to include list_rates and sum_ratesand neuon_active_neighbors, graph, indegree function etc
function update(system::ContactProcess, n::Int)::Int
    rate_n_old = system.rates[n];
    dstate     = 0;
    if system.neurons[n] == 0  
        system.neurons[n] = 1;
        system.rates[n]   = system.mu;
        dstate            = 1;
    else  
        system.neurons[n] = 0;
        system.rates[n]   = rate_activate(system, n)
        dstate            = -1;
    end

    # update rate list
    system.sum_rates += (system.rates[n] - rate_n_old);
    for nn in system.outgoing_neighbors(n)
        system.active_incoming_neighbors[nn] += dstate;
        if system.neurons[nn] == 0  
            rate_nn_old       = system.rates[nn];
            system.rates[nn]  = rate_activate(system, nn)
            system.sum_rates += (system.rates[nn] - rate_nn_old); 
        end
    end

    return dstate
end

@inline function rate_activate(system::ContactProcess, n::Int)::Float64
    return system.h + system.lambda * system.active_incoming_neighbors[n] / length(system.incoming_neighbors(n)) 
end

##############################run
# avg_activity, avg_activity2, T = run(Int(1e3),1.0,1e-3,1e-2,1e6,1e3,rng)
# println(avg_activity, " ", avg_activity2, " ", T)

# err = sqrt((avg_activity2-avg_activity*avg_activity)/(T-1))
# println(avg_activity, " ", err, " ", T)


