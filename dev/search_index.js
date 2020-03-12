var documenterSearchIndex = {"docs":
[{"location":"non_equilibrium/#Kintetic-Monte-Carlo-1","page":"Non-Equilibrium Tools","title":"Kintetic Monte-Carlo","text":"","category":"section"},{"location":"non_equilibrium/#","page":"Non-Equilibrium Tools","title":"Non-Equilibrium Tools","text":"next\nnext_time\nnext_event\nadvance!","category":"page"},{"location":"non_equilibrium/#MonteCarloX.next","page":"Non-Equilibrium Tools","title":"MonteCarloX.next","text":"next(alg::KineticMonteCarlo, [rng::AbstractRNG,] rates::AbstractWeights)::Tuple{Float64,Int}\n\nNext stochastic event (\\Delta t, index) drawn proportional to probability given in rates\n\n\n\n\n\nnext(alg::KineticMonteCarlo, [rng::AbstractRNG,] event_handler::AbstractEventHandlerRate)\n\nNext stochastic event (\\Delta t, event type) organized by event_handler fast(to be tested, depends on overhead of EventList) implementation of nexteventrate if defined by EventList object\n\n\n\n\n\nGenerate a new event from a collection of inhomogeneous poisson processes with rates Lambda(t).\n\nArguments\n\nrates: rates(dt); Float -> [Float]\nmax_rate: maximal rate in near future (has to be evaluated externally)\nrng: random number generator\n\nAPI - output\n\nreturns the next event time and event id as tuple (dt, id)\n\n\n\n\n\n","category":"function"},{"location":"non_equilibrium/#MonteCarloX.next_time","page":"Non-Equilibrium Tools","title":"MonteCarloX.next_time","text":"next_time([rng::AbstractRNG,] rate::Float64)::Float64\n\nNext stochastic \\Delta t for Poisson process with rate\n\n\n\n\n\nnexteventtime(rate::Function, max_rate::Float64, rng::AbstractRNG)::Float64\n\nGenerate a new event from an inhomogeneous poisson process with rate Lambda(t). Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  2) see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf\n\nArguments\n\nrate: rate(dt) has to be defined outside (e..g t-> rate(t+t0,args))\nmax_rate: maximal rate in near future (has to be evaluated externally)\nrng: random number generator\n\nAPI - output\n\nreturns the next event time\n\n\n\n\n\nGenerate a new event from an inhomogeneous poisson process with rate Lambda(t) under the assumption that rate(dt) is monotonically decreasing. Based on (Ogata’s Modified Thinning Algorithm: Ogata,  1981,  p.25,  Algorithm  3) see also https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf\n\nArguments\n\nrate: rate(dt) has to be defined outside (e..g t-> rate(t+t0,args))\nrng: random number generator\n\nAPI - output\n\nreturns the next event time\n\n\n\n\n\n","category":"function"},{"location":"non_equilibrium/#MonteCarloX.next_event","page":"Non-Equilibrium Tools","title":"MonteCarloX.next_event","text":"next_event(rng::AbstractRNG, cumulated_rates::Vector{T})::Int where {T<:AbstractFloat}\n\nSelect a single random index in 1:length(cumulated_rates) with cumulated probability given in cumulated_rates.\n\nRemarks\n\nDeprecated unless we find a good data structure for (dynamic) cumulated rates\n\n\n\n\n\nnext_event([rng::AbstractRNG,] rates::AbstractWeights)::Int\n\nSelect a single random index in 1:length(rates) with probability proportional to the entry in rates.\n\nRemarks\n\nThis is on average twice as fast as StatsBase.sampling because it can iterate from either beginning or end of rates\n\n\n\n\n\nnext_event([rng::AbstractRNG,] event_handler::AbstractEventHandlerRate{T})::T where T\n\nSelect a single random event with a given probability managed by event_handler.\n\nThe event_handler also manages the case that no valid events are left (e.g. when all rates are equal to zero). This becomes relevant when using advance! to advance for some time.\n\nSee also: advance!\n\n\n\n\n\nGenerate a new event id from a collection of inhomogeneous poisson processes with rates Lambda(t).\n\nArguments\n\nrates: rates(dt); Float -> [Float]\nmax_rate: maximal rate in near future (has to be evaluated externally)\nrng: random number generator\n\nAPI - output\n\nreturns the next event id\n\n\n\n\n\n","category":"function"},{"location":"non_equilibrium/#MonteCarloX.advance!","page":"Non-Equilibrium Tools","title":"MonteCarloX.advance!","text":"advance!(alg::KineticMonteCarlo, [rng::AbstractRNG], event_handler::AbstractEventHandlerRate, update!::Function, total_time::T)::T where {T<:Real}\n\nDraw events from event_handler and update event_handler with update! until total_time has passed. Return time of last event.\n\n\n\n\n\n","category":"function"},{"location":"non_equilibrium/#Inhomogeneous-Poisson-Process-1","page":"Non-Equilibrium Tools","title":"Inhomogeneous Poisson Process","text":"","category":"section"},{"location":"non_equilibrium/#","page":"Non-Equilibrium Tools","title":"Non-Equilibrium Tools","text":"","category":"page"},{"location":"equilibrium/#Importance-Sampling-(Metropolis)-1","page":"Equilibrium Tools","title":"Importance Sampling (Metropolis)","text":"","category":"section"},{"location":"equilibrium/#","page":"Equilibrium Tools","title":"Equilibrium Tools","text":"accept\nsweep","category":"page"},{"location":"equilibrium/#MonteCarloX.accept","page":"Equilibrium Tools","title":"MonteCarloX.accept","text":"accept(log_weight::Function, args_new::Tuple{Number, N}, args_old::Tuple{Number, N}, rng::AbstractRNG)::Bool where N\n\nEvaluate most general acceptance probability for imporance sampling of P(x) propto e^log_weight(x).\n\nArguments\n\nlog_weight(args): logarithmic ensemble weight function, e.g., canomical ensemble log_weight(x) = -beta x\nargs_new: arguments (can be Number or Tuple) for new (proposed) state\nargs_old: arguments (can be Number or Tuple) for old                        state\nrng: random number generator, e.g. MersenneTwister\n\nSpecializations\n\naccept(alg::Metropolis(), rng::AbstractRNG, xnew::T, xold::T) where T (@ref)\n\n\n\n\n\naccept(alg::Metropolis, rng::AbstractRNG, beta::Float64, dx::T)::Bool where T\n\nStandard metropolis algorithm with  p(x\\to x^') = \\text{min}\\left(1, e^{-\\beta \\Delta x}\\right) where dx = x^' - x\n\n\n\n\n\naccept(alg::MetropolisHastings, rng::AbstractRNG, beta::Float64, dx::T, q_old::Float64, q_new::Float64)::Bool where T\n\n\n\n\n\n","category":"function"},{"location":"equilibrium/#MonteCarloX.sweep","page":"Equilibrium Tools","title":"MonteCarloX.sweep","text":"sweep(list_updates, list_weights::AbstractWeights, rng::AbstractRNG; number_updates::Int=1) where T<:AbstractFloat\n\nRandomly pick und run update (has to check acceptance by itself!) from list_updates with probability specified in list_probabilities and repeat this number_updates times.\n\n\n\n\n\n","category":"function"},{"location":"equilibrium/#Reweighting-1","page":"Equilibrium Tools","title":"Reweighting","text":"","category":"section"},{"location":"equilibrium/#","page":"Equilibrium Tools","title":"Equilibrium Tools","text":"","category":"page"},{"location":"#MonteCarloX-1","page":"Getting Started","title":"MonteCarloX","text":"","category":"section"},{"location":"#","page":"Getting Started","title":"Getting Started","text":"The goal of MonteCarloX.jl is to offer a framework with standard and advanced Monte Carlo tools for equilibrium and non-equilibrium simulations in Julia. ","category":"page"},{"location":"helper/#Helper-Tools-1","page":"Helper","title":"Helper Tools","text":"","category":"section"},{"location":"helper/#","page":"Helper","title":"Helper","text":"Order = [:type, :function]\nPages = [\"helper.md\"]","category":"page"},{"location":"helper/#Full-Docs-1","page":"Helper","title":"Full Docs","text":"","category":"section"},{"location":"helper/#","page":"Helper","title":"Helper","text":"Modules = [MonteCarloX]\nPages   = [\n    \"utils.jl\",\n]\nPrivate = false","category":"page"},{"location":"helper/#MonteCarloX.binary_search-Union{Tuple{T}, Tuple{AbstractArray{T,1},T}} where T<:Real","page":"Helper","title":"MonteCarloX.binary_search","text":"binary_search(sorted::AbstractVector{T}, value::T)::Int where {T<:Real}\n\nPerfom a binary search to return the index i of an sorted array such that sorted[i-1] < value <= sorted[i]\n\nExamples\n\njulia> MonteCarloX.binary_search([1.,2.,3.,4.],2.5)\n3\n\njulia> MonteCarloX.binary_search([1,2,3,4],2)\n2\n\n\n\n\n\n","category":"method"},{"location":"helper/#MonteCarloX.log_sum-Union{Tuple{T}, Tuple{T,T}} where T<:AbstractFloat","page":"Helper","title":"MonteCarloX.log_sum","text":"log_sum(a::T,b::T)\n\nReturn result of logarithmic sum c = ln(A+B) = a + ln(1+e^b-a) where C = e^c = A+B = e^a + e^b. \n\nThis is useful for sums that involve elements that span multiple orders of magnitude, e.g., the partition sum that is required as normalization factor during reweighting.\n\nExamples\n\njulia> exp(MonteCarloX.log_sum(log(2.), log(3.)))\n5.000000000000001\n\n\n\n\n\n\n","category":"method"},{"location":"helper/#MonteCarloX.random_element-Tuple{Array{Float64,1},Random.AbstractRNG}","page":"Helper","title":"MonteCarloX.random_element","text":"random_element(list_probabilities::Vector{T},rng::AbstractRNG)::Int where T<:AbstractFloat\n\nPick an index with probability defined by list_probability (which needs to be normalized). \n\n#Remark Deprecated for use of StatsBase.sample\n\nExamples\n\njulia> using Random\n\njulia> rng = MersenneTwister(1000);\n\njulia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)\n4\njulia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)\n4\njulia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)\n3\njulia> MonteCarloX.random_element([0.1,0.2,0.3,0.4],rng)\n4\n\n\n\n\n\n","category":"method"}]
}
