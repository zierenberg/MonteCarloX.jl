#TODO: example with large deviation simulation
#TODO: update example of contact process with the implementation from sonnet
using MonteCarloX
using StatsBase
using Random
using LightGraphs
using Plots

"""
main program that implements full example simulation of 2D Ising model and compares to the Beale solution
call this from within julia -O3
@time main(flag_plot=false) should run O(4s) comparable with C code from Martin?!
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

    println("Multicanonical initialization")
    alg = Multicanonical()
    E_max =  8*8
    E_min = -E_max
    dE = 4
    log_weight = Histogram(E_min:dE:E_max) 
    histogram  = zero(log_weight)

    println("Multicanonical iteration")
    for i=1:100
      histogram.weights .= 0
      run_iteration(rng, system, log_weight, histogram, Int(1e5), 1000, rng)
      update_weights(alg, log_weight, histogram)
    end
    
    #run_production(rng, system, log_weight, histogram, Int(1e5), 1000, rng)
    #list_est_E = reweight()...
    ## plot energy compared to Beale
    #if flag_plot
    #    display(plot!(list_beta, list_est_E / 64.0, yerr = list_err_E / 64.0, seriestype = :scatter, label = "MC", subplot = 1))
    #    display(plot!(list_beta, list_est_C / 64.0, yerr = list_err_C / 64.0, seriestype = :scatter, label = "MC", subplot = 2))
    #end
end



"""
run simulation
"""
function run_iteration(rng::AbstractRNG, system::IsingSystem, log_weight::Histogram, histogram::Histogram, n_meas::Int64, n_therm::Int64)
    current_E = E(system)
    N = length(system.spins)
    for sweep in 1:n_meas + n_therm
        # sweep
        for step in 1:N
            dE = update_spin_flip(rng, system, log_weight, current_E)
            current_E += dE
            push!(histogram,current_E)
        end
        #current_E = E(system)
    end
end

function update_spin_flip(rng::AbstractRNG, system::IsingSystem, log_weight::Histogram, E_old::Int)::Int
    index = rand(rng, 1:length(system.spins))
    dE    = -2 * E_local(system, index)
    if accept(Multicanonical(), rng, log_weight, E_new+dE, E_old)
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



###########################################################################
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

# logarithmic density of states for the 2D 8x8 Ising model from Bealse solution
log_dos_beale_8x8 = [(-128, 0.6931471805599453), (-120, 4.852030263919617), (-116, 5.545177444479562), (-112, 8.449342524508063), (-108, 9.793672686528922), (-104, 11.887298863200714), (-100, 13.477180596840947), (-96, 15.268195474147658), (-92, 16.912371686315282), (-88, 18.59085846191256), (-84, 20.230089202801466), (-80, 21.870810400320693), (-76, 23.498562234123614), (-72, 25.114602234581373), (-68, 26.70699035290573), (-64, 28.266152815389898), (-60, 29.780704423363996), (-56, 31.241053997806176), (-52, 32.63856452513369), (-48, 33.96613536105969), (-44, 35.217576663643314), (-40, 36.3873411250109), (-36, 37.47007844691906), (-32, 38.46041522581422), (-28, 39.35282710786369), (-24, 40.141667825183845), (-20, 40.82130289691285), (-16, 41.38631975325592), (-12, 41.831753810069756), (-8, 42.153328313883975), (-4, 42.34770636939425), (0, 42.41274640460084), (4, 42.34770636939425), (8, 42.153328313883975), (12, 41.831753810069756), (16, 41.38631975325592), (20, 40.82130289691285), (24, 40.141667825183845), (28, 39.35282710786369), (32, 38.46041522581422), (36, 37.47007844691906), (40, 36.3873411250109), (44, 35.217576663643314), (48, 33.96613536105969), (52, 32.63856452513369), (56, 31.241053997806176), (60, 29.780704423363996), (64, 28.266152815389898), (68, 26.70699035290573), (72, 25.114602234581373), (76, 23.498562234123614), (80, 21.870810400320693), (84, 20.230089202801466), (88, 18.59085846191256), (92, 16.912371686315282), (96, 15.268195474147658), (100, 13.477180596840947), (104, 11.887298863200714), (108, 9.793672686528922), (112, 8.449342524508063), (116, 5.545177444479562), (120, 4.852030263919617), (128, 0.6931471805599453)];
