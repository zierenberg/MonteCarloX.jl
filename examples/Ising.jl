using LightGraphs
using MonteCarloX
using Random
using Plots

# logarithmic density of states for the 2D 8x8 Ising model from Bealse solution
log_dos_beale_8x8 = [(-128, 0.6931471805599453), (-120, 4.852030263919617), (-116, 5.545177444479562), (-112, 8.449342524508063), (-108, 9.793672686528922), (-104, 11.887298863200714), (-100, 13.477180596840947), (-96, 15.268195474147658), (-92, 16.912371686315282), (-88, 18.59085846191256), (-84, 20.230089202801466), (-80, 21.870810400320693), (-76, 23.498562234123614), (-72, 25.114602234581373), (-68, 26.70699035290573), (-64, 28.266152815389898), (-60, 29.780704423363996), (-56, 31.241053997806176), (-52, 32.63856452513369), (-48, 33.96613536105969), (-44, 35.217576663643314), (-40, 36.3873411250109), (-36, 37.47007844691906), (-32, 38.46041522581422), (-28, 39.35282710786369), (-24, 40.141667825183845), (-20, 40.82130289691285), (-16, 41.38631975325592), (-12, 41.831753810069756), (-8, 42.153328313883975), (-4, 42.34770636939425), (0, 42.41274640460084), (4, 42.34770636939425), (8, 42.153328313883975), (12, 41.831753810069756), (16, 41.38631975325592), (20, 40.82130289691285), (24, 40.141667825183845), (28, 39.35282710786369), (32, 38.46041522581422), (36, 37.47007844691906), (40, 36.3873411250109), (44, 35.217576663643314), (48, 33.96613536105969), (52, 32.63856452513369), (56, 31.241053997806176), (60, 29.780704423363996), (64, 28.266152815389898), (68, 26.70699035290573), (72, 25.114602234581373), (76, 23.498562234123614), (80, 21.870810400320693), (84, 20.230089202801466), (88, 18.59085846191256), (92, 16.912371686315282), (96, 15.268195474147658), (100, 13.477180596840947), (104, 11.887298863200714), (108, 9.793672686528922), (112, 8.449342524508063), (116, 5.545177444479562), (120, 4.852030263919617), (128, 0.6931471805599453)];

struct IsingSystem{F}
    dims::Vector{Int}
    lattice::SimpleGraph
    spins::Vector{Int}
    nearest_neighbors::F
end

function constructIsing(dims::Vector{Int}, rng::AbstractRNG)::IsingSystem
    lattice = LightGraphs.SimpleGraphs.grid(dims, periodic = true)
    spins   = rand(rng, [-1,1], nv(lattice))
    nearest_neighbors(i) = outneighbors(lattice, i)
    return IsingSystem(dims, lattice, spins, nearest_neighbors)
end

function M(system::IsingSystem)::Int
    return abs(sum(system.spins))
end

function analytic(beta::Float64, log_dos::Vector{Tuple{Int,Float64}})::Tuple{Float64,Float64}
    mean_E  = 0.0
    mean_E2 = 0.0
    norm    = 0.0
    for (E, log_d) in log_dos
        mean_E  +=   E * exp(log_d - beta * E)
        mean_E2 += E^2 * exp(log_d - beta * E)
        norm    +=     exp(log_d - beta * E)
    end
    mean_E  /= norm
    mean_E2 /= norm
    return mean_E, beta^2 * (mean_E2 - mean_E^2)
end

# J=1
# E=-J*sum s_i s_j
function E(system::IsingSystem)::Int
    E = 0
    for i in 1:length(system.spins)
        E += E_local(system, i)
    end
    return E / 2
end

function E_local(system::IsingSystem, index::Int)::Int
    e = 0
    for j in system.nearest_neighbors(index)
        e -= system.spins[index] * system.spins[j]
    end
    return e
end

"""
run simulation
"""
function run(system::IsingSystem, beta::Float64, n_meas::Int64, n_therm::Int64, rng::AbstractRNG)
    current_E      = E(system)
    measurements_E = zeros(n_meas)
    measurements_M = zeros(n_meas)
    # histogram_E = Dict([(E,0) for (E,log_d) in log_dos_beale_8x8])
    N = length(system.spins)
    for sweep in 1:n_meas + n_therm
        # Metropolis sweep
        for step in 1:N
            dE = update_spin_flip(system, beta, rng)
            current_E += dE
        end
        # Cluster update
        ClusterWolff.update(system.spins, system.nearest_neighbors, beta, rng)
        current_E = E(system)
        if sweep > n_therm
            current_M = M(system)
            measurements_E[sweep - n_therm] = current_E
            measurements_M[sweep - n_therm] = current_M
        end
    end

    return measurements_E, measurements_M

end

function update_spin_flip(system::IsingSystem, beta::Float64, rng::AbstractRNG)::Int
    # define weight function via energy change and simply pass 0 as second argument
    log_weight(dE::Int)::Float64 = -beta * dE

    index = rand(rng, 1:length(system.spins))
    dE    = -2 * E_local(system, index)
    if Metropolis.accept(log_weight, dE, 0, rng)
        system.spins[index] *= -1
    else
        dE = 0
    end
    return dE
end

function mean_err(X::Vector{Float64})::Tuple{Float64,Float64}
    mean  = 0.0
    mean2 = 0.0
    for x in X
        mean  += x
        mean2 += x * x
    end
    N = length(X)
    mean  /= N
    mean2 /= N
    return mean, sqrt((mean2 - mean) / (N - 1))
end

"""
main program that implements full example simulation of 2D Ising model and compares to the Beale solution
"""
function main(;flag_plot::Bool = true)
    println("Initialize system (8x8)")
    rng = MersenneTwister(1000)
    system = constructIsing([8,8], rng)

    if flag_plot
        p_E = plot(xaxis = ("beta"), yaxis = ("energy per spin"))
        p_C = plot(xaxis = ("beta"), yaxis = ("specific heat"))
        display(plot(p_E, p_C, layout = (2, 1)))
    end

    # eval Beale solution
    println("Evaluate exact solution")
    ana_beta = [beta for beta in 0.0:0.001:0.8]
    ana_E = []
    ana_C = []
    for beta in ana_beta
        E, C  = analytic(beta, log_dos_beale_8x8)
        push!(ana_E, E)
        push!(ana_C, C)
    end
    if flag_plot
        display(plot!(ana_beta, ana_E / 64.0, label = "Beale", subplot = 1))
        display(plot!(ana_beta, ana_C / 64.0, label = "Beale", subplot = 2))
    end

    # plot()
    # loop over beta
    println("Monte Carlo simulation")
    list_beta = [beta for beta in 0.0:0.1:0.7]
    # list_beta = [beta for beta in 0.0:0.1:0.3]
    list_est_E = zeros(length(list_beta))
    list_err_E = zeros(length(list_beta))
    list_est_C = zeros(length(list_beta))
    list_err_C = zeros(length(list_beta))
    # thermalization
    run(system, 0.0, 0, 1000, rng)
    for (i, beta) in enumerate(list_beta)
        measurements_E, measurements_M = run(system, beta, Int(1e5), 1000, rng)
        est_E, err_E = mean_err(measurements_E)
        measurements_C = beta^2 * (measurements_E.^2 - ones(length(measurements_E)) * est_E^2)
        est_C, err_C = mean_err(measurements_C)
        est_M, err_M = mean_err(measurements_M)
        list_est_E[i] = est_E
        list_err_E[i] = err_E
        list_est_C[i] = est_C
        list_err_C[i] = err_C
        println(beta, " ", est_E, " ", err_E, " ", est_C, " ", err_C, " ", est_M, " ", err_M)
        # E_base = sort(collect(keys(histogram_E)))
        # display(plot!(E_base,[histogram_E[E] for E in E_base], label=beta))
    end

    # plot energy compared to Beale
    if flag_plot
        display(plot!(list_beta, list_est_E / 64.0, yerr = list_err_E / 64.0, seriestype = :scatter, label = "MC", subplot = 1))
        display(plot!(list_beta, list_est_C / 64.0, yerr = list_err_C / 64.0, seriestype = :scatter, label = "MC", subplot = 2))
    end
end

# main()
