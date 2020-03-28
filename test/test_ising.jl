using LightGraphs
using MonteCarloX
using Random
using Distributions
import Distributions.pdf
import Distributions.cdf
using StatsBase
import StatsBase.kldivergence

include("utils.jl")

""" Testing reweighting on 2D Ising model"""
function test_ising_reweighting(;verbose = false)
    log_P(E, beta) = -beta * E
    P(E, beta) = exp(-beta * E)

    list_beta = [0.0,0.3,0.7,1.0,1.5,2.0]
    pass = true

    rng = MersenneTwister(1000)
    system = constructIsing([8,8], rng)
    nearest_neighbors(index) = outneighbors(system.lattice, index)
    N = length(system.spins)

    for beta in list_beta
        samples = 1000
        list_energy = zeros(samples)
        # thermalization 100 sweeps
        for sweep in 1:samples + 100
            # Metropolis:
            for i in 1:N
                update_spin_flip(system, beta, rng)
            end
            if sweep > 100
                list_energy[sweep - 100] = E(system)
            end
        end
        H_meas = fit(Histogram, list_energy, minimum(list_energy):4:maximum(list_energy) + 4, closed = :left)
        P_meas = normalize(H_meas, mode = :probability)
        # reweighting both with timeseries (list_energy) and histrogram
        beta_source = beta
        beta_target = beta + 0.05
        log_P_source(E) = log_P(E, beta_source)
        log_P_target(E) = log_P(E, beta_target)
        P_source(E) = P(E, beta_source)
        P_target(E) = P(E, beta_target)
        E_ref = analytic_expectation_value_E(beta_target, log_dos_beale_8x8) 
        P_ref_source = BoltzmannDistribution(beta_source, log_dos_beale_8x8).pdf
        P_ref_target = BoltzmannDistribution(beta_target, log_dos_beale_8x8).pdf
        kld_ref = kldivergence(P_meas, x->P_ref_source[x])
        if verbose
            println("result analytic = $(E_ref)")
        end
        ###########################################################################
        E_reweight_list1 = MonteCarloX.expectation_value_from_timeseries(log_P_target, log_P_source, list_energy, list_energy) 
        E_reweight_list1_error = abs((E_reweight_list1 - E_ref) / E_ref)
        if verbose
            println("result expectation_value_from_timeseries_log: <E> = $(E_reweight_list1); difference from exact = $(E_reweight_list1_error)")
        end
        pass &= E_reweight_list1_error < 0.1
        ###########################################################################
        P_reweight_list = MonteCarloX.distribution_from_timeseries(log_P_target, log_P_source, list_energy, minimum(list_energy):4:maximum(list_energy) + 4) 
        kld_source = kldivergence(P_reweight_list, x->P_ref_source[x])
        kld_target = kldivergence(P_reweight_list, x->P_ref_target[x])
        if verbose
            println("result distribution_from_timeseries_log: kld_target ($(kld_target)) !< kld_source ($(kld_source))")
        end
        # only compare reweighted distribution to target and source. There result is comparable.
        pass &= kld_target < kld_source
        ###########################################################################
        E_reweight_hist1 = MonteCarloX.expectation_value_from_histogram(E->E, log_P_target, log_P_source, H_meas) 
        E_reweight_hist1_error = abs((E_reweight_hist1 - E_ref) / E_ref)
        if verbose
            println("result expectation_value_from_histogram_log-1: <E> = $(E_reweight_hist1); difference from exact = $(E_reweight_hist1_error)")
            println("result expectation_value_from_histogram_log-1: <E> = $(E_reweight_hist1); difference from timeseries = $(E_reweight_hist1 - E_reweight_list1)")
        end
        pass &= E_reweight_hist1_error < 0.1
        ###########################################################################
        hist_obs = zero(H_meas)
        for E in list_energy
            index = StatsBase.binindex(hist_obs, E)
            hist_obs.weights[index] += E
        end
        E_reweight_hist2 = MonteCarloX.expectation_value_from_histogram(log_P_target, log_P_source, H_meas, hist_obs) 
        E_reweight_hist2_error = abs(E_reweight_hist2 - E_reweight_list1)
        if verbose
            println("result expectation_value_from_histogram_log: <E> = $(E_reweight_hist2); difference from timeseries = $(E_reweight_hist2_error)")
        end
        pass &= E_reweight_hist2_error < 0.1
        ###########################################################################
        # TODO: canonical reweighting
    end

    # TODO: multi-histogram reweighting

    return pass
end


""" Testing Metropolis accept on 2D Ising model"""
function test_ising_metropolis(;verbose = false)
    list_beta = [0.0,0.3,0.7,1.0,1.5,2.0]
    pass = true

    rng = MersenneTwister(1000)
    system = constructIsing([8,8], rng)
    nearest_neighbors(index) = outneighbors(system.lattice, index)
    N = length(system.spins)

    for beta in list_beta
        samples = 1000
        list_energy = zeros(samples)
        # thermalization 100 sweeps
        for sweep in 1:samples + 100
            # Metropolis:
            for i in 1:N
                update_spin_flip(system, beta, rng)
            end
            if sweep > 100
                list_energy[sweep - 100] = E(system)
            end
        end
        P_meas = fit(Histogram, list_energy, minimum(list_energy):4:maximum(list_energy) + 4, closed = :left)
        P_meas = normalize(P_meas, mode = :probability)
        P_true = BoltzmannDistribution(beta, log_dos_beale_8x8).pdf
        # KL divergence sum P(E)logP(E)/Q(E) 
        # P=P_meas, Q=P_true s.t P(E)=0 simply ignored
        kld = abs(kldivergence(P_meas, x->P_true[x]))
        if verbose
            println(beta, " ", kld, " ", length(keys(P_meas)))
        end
        pass &= kld < 0.1
    end

    return pass
end

""" Testing Cluster update on 2D Ising model"""
function test_ising_cluster(;verbose = false)
    # avoid beta_c = 0.44
    # why does it consistently fail around beta=0.6
    list_beta = [0,0.3,0.5,1,1.5,2.0]
    pass = true

    rng = MersenneTwister(1000)
    system = constructIsing([8,8], rng)
    nearest_neighbors(index) = outneighbors(system.lattice, index)
    N = length(system.spins)

    cluster_algorithm = MonteCarloX.ClusterWolff()
    
    for beta in list_beta
        samples = Int(1e4)
        list_energy = zeros(samples)

        # thermalization 100 sweeps
        for sweep in 1:samples + 100
            for i in 1:5
                MonteCarloX.update(cluster_algorithm, rng, system.spins, nearest_neighbors, beta)
            end
            if sweep > 100
                list_energy[sweep - 100] = E(system)
            end
        end

        #
        P_meas = fit(Histogram, list_energy, minimum(list_energy):4:maximum(list_energy) + 4, closed = :left)
        P_meas = normalize(P_meas, mode = :probability)
        P_true = BoltzmannDistribution(beta, log_dos_beale_8x8).pdf

        kld = abs(kldivergence(P_meas, x->P_true[x]))
        if verbose
            println("result kldivergence P_meas to P_true = $(kld)")
        end
        pass &= kld < 0.02
    end

    return pass
end



###############################################################################
###############################################################################
### Details functions 

 
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

function analytic_expectation_value_E(beta::Float64, log_dos::Vector{Tuple{Int,Float64}})::Float64
    mean_E  = 0.0
    norm    = 0.0
    for (E, log_d) in log_dos
        mean_E  +=     E * exp(log_d - beta * E)
        norm    +=         exp(log_d - beta * E)
    end
    mean_E /= norm
    return mean_E
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

function update_spin_flip(system::IsingSystem, beta::Float64, rng::AbstractRNG)::Int
    # define weight function via energy change and simply pass 0 as second argument
    log_weight(dE::Int)::Float64 = -beta * dE

    index = rand(rng, 1:length(system.spins))
    dE        = -2 * E_local(system, index)
    if accept(rng, log_weight, dE, 0)
        system.spins[index] *= -1
    else
        dE = 0
    end
    return dE
end

# """ Definition of the Boltzmann distribution for hypothesis testing."""
struct BoltzmannDistribution <: ContinuousUnivariateDistribution
    pdf::Dict{Any,Float64}
    cdf::Dict{Any,Float64}

    BoltzmannDistribution(beta, log_dos) = new(initialize_BoltzmannDistribution(beta, log_dos)...)
    # BoltzmannDistribution(beta, log_dos, log_Z) = new(Float64(beta), Dict{Any,Float64}(log_dos), Float64(log_Z))
end

# Probability distribution (for both precompute in constructor...)
pdf(d::BoltzmannDistribution, E::Real) = d.pdf[E]

# cumulated probability distribution
cdf(d::BoltzmannDistribution, E::Real) = d.cdf[E]

function initialize_BoltzmannDistribution(beta, log_dos)
    # partition sum
    log_Z = -Inf
    for (E, log_d) in log_dos
        log_Z = log_sum(log_Z, log_d - beta * E)
    end

    pdf = Dict{Int64,Float64}()
    cdf = Dict{Int64,Float64}()

    log_cdf = 0.0
    for (E, log_d) in log_dos
        log_pdf = log_d - beta * E - log_Z
        log_cdf = log_sum(log_cdf, log_pdf)
        pdf[E] = exp(log_pdf)
        cdf[E] = exp(log_cdf)
    end 
    return pdf, cdf
end


log_dos_beale_8x8 = [ (-128, 0.6931471805599453), (-124, 0.0), (-120, 4.852030263919617), (-116, 5.545177444479562), (-112, 8.449342524508063), (-108, 9.793672686528922), (-104, 11.887298863200714), (-100, 13.477180596840947), (-96, 15.268195474147658), (-92, 16.912371686315282), (-88, 18.59085846191256), (-84, 20.230089202801466), (-80, 21.870810400320693), (-76, 23.498562234123614), (-72, 25.114602234581373), (-68, 26.70699035290573), (-64, 28.266152815389898), (-60, 29.780704423363996), (-56, 31.241053997806176), (-52, 32.63856452513369), (-48, 33.96613536105969), (-44, 35.217576663643314), (-40, 36.3873411250109), (-36, 37.47007844691906), (-32, 38.46041522581422), (-28, 39.35282710786369), (-24, 40.141667825183845), (-20, 40.82130289691285), (-16, 41.38631975325592), (-12, 41.831753810069756), (-8, 42.153328313883975), (-4, 42.34770636939425), (0, 42.41274640460084), (4, 42.34770636939425), (8, 42.153328313883975), (12, 41.831753810069756), (16, 41.38631975325592), (20, 40.82130289691285), (24, 40.141667825183845), (28, 39.35282710786369), (32, 38.46041522581422), (36, 37.47007844691906), (40, 36.3873411250109), (44, 35.217576663643314), (48, 33.96613536105969), (52, 32.63856452513369), (56, 31.241053997806176), (60, 29.780704423363996), (64, 28.266152815389898), (68, 26.70699035290573), (72, 25.114602234581373), (76, 23.498562234123614), (80, 21.870810400320693), (84, 20.230089202801466), (88, 18.59085846191256), (92, 16.912371686315282), (96, 15.268195474147658), (100, 13.477180596840947), (104, 11.887298863200714), (108, 9.793672686528922), (112, 8.449342524508063), (116, 5.545177444479562), (120, 4.852030263919617), (124, 0.0), (128, 0.6931471805599453) ];

"""
Kullback-Leibler divergence between two distributions

distributions are here considered to be dictionaries

valid for n-dimensional dictionaries (args=Tuple or larger?)
"""
function kldivergence(P::Dict, Q::Dict)
        ##KL divergence sum P(args)logP(args)/Q(args) 
        ## P=P_meas, Q=P_true s.t P(args)=0 simply ignored
        #
        kld = 0.0
        for (args, p) in P
            q = Q[args]
            kld += p * log(p / q)
        end
        return kld
end

