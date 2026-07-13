# # Reversible Dimerization with Gillespie Algorithm
#
# This example demonstrates continuous-time stochastic simulation of a
# reversible dimerization reaction using the Gillespie algorithm:
#
# ```math
# A + B \underset{k_\text{off}}{\stackrel{k_\text{on}}{\rightleftharpoons}} AB
# ```
#
# The Gillespie algorithm samples exact trajectories of the chemical master
# equation by drawing the time to the next reaction event from an exponential
# distribution and selecting the reaction channel proportionally to its rate.
#
# We implement the same simulation twice: first **handwritten**, where the
# propensities are handed to `step!` as a plain rate function, and then with
# the core **`ReactionEvents`** generator, which maintains the propensities in
# a persistent ledger behind the same loop.

using Random, StatsBase, Plots
using MonteCarloX

# ## System definition
#
# The system holds only the molecule counts and rate constants — pure state, no
# rate bookkeeping. Both versions share it: `modify!` fires a reaction channel
# (association A + B → AB, dissociation AB → A + B).

mutable struct ReversibleDimerModel <: AbstractSystem
    A    :: Int
    B    :: Int
    AB   :: Int
    k_on  :: Float64
    k_off :: Float64
end

function MonteCarloX.modify!(sys::ReversibleDimerModel, event::Int, t)
    if event == 1 && sys.A > 0 && sys.B > 0
        sys.A -= 1;  sys.B -= 1;  sys.AB += 1
    elseif event == 2 && sys.AB > 0
        sys.A += 1;  sys.B += 1;  sys.AB -= 1
    end
    return sys
end

# ## Parameters
#
# We start with 30 molecules of A, 20 of B, and no dimers. The on-rate
# ``k_\text{on} = 0.01`` and off-rate ``k_\text{off} = 0.5`` set the
# equilibrium dimer fraction. Both runs record the counts every 0.5 time units.

T = 200.0
measurement_times = collect(0.0:0.5:T)
make_measurements() = Measurements([
    :A  => (s -> s.A)  => Int[],
    :B  => (s -> s.B)  => Int[],
    :AB => (s -> s.AB) => Int[],
], measurement_times)

# ## Version 1: handwritten rate function
#
# The propensities are a plain function of the state, rebuilt from scratch at
# every step and handed to `step!` as a rate function of time (a `Function` is
# a valid event source; the time argument is unused here but allows explicitly
# time-dependent rates). Nothing is maintained between events — at the cost of
# evaluating and allocating the full propensity vector for every single one.
# Measurements are recorded **before** `modify!` fires the channel, so each
# sample reflects the state during the inter-event interval.

## propensities: rates at which each reaction fires
reaction_rates(sys::ReversibleDimerModel, t) = [
    sys.k_on  * sys.A * sys.B,   ## association
    sys.k_off * sys.AB,          ## dissociation
]

sys1  = ReversibleDimerModel(30, 20, 0, 0.01, 0.5)
alg1  = Gillespie(MersenneTwister(23))
meas1 = make_measurements()

measure!(meas1, sys1, alg1.time)        ## record initial state
while alg1.time <= T
    t_new, event = step!(alg1, t -> reaction_rates(sys1, t))
    measure!(meas1, sys1, t_new)        ## before modify!
    modify!(sys1, event, t_new)         ## fire the channel
end

println("Handwritten : A=$(sys1.A), B=$(sys1.B), AB=$(sys1.AB) after $(alg1.steps) events")

# ## Version 2: the `ReactionEvents` generator
#
# `ReactionEvents` maintains the propensities in a persistent ledger instead of
# rebuilding them per event. The system plugs in through the reaction
# interface — `nreactions` plus a per-channel rate rule — and the generator's
# *own* `modify!` fires the channel on the system and re-evaluates the rule.
# The loop is the same standard KMC pair `step!`/`modify!`.

MonteCarloX.nreactions(::ReversibleDimerModel) = 2

## mass action again, now as a per-channel rule
propensity(sys::ReversibleDimerModel, r::Int) =
    r == 1 ? sys.k_on * sys.A * sys.B : sys.k_off * sys.AB

sys2  = ReversibleDimerModel(30, 20, 0, 0.01, 0.5)
src   = ReactionEvents(sys2, propensity)
alg2  = Gillespie(MersenneTwister(23))
meas2 = make_measurements()

measure!(meas2, sys2, alg2.time)
while alg2.time <= T
    t_new, event = step!(alg2, src)
    measure!(meas2, sys2, t_new)
    modify!(src, event, t_new)
end

println("Generator   : A=$(sys2.A), B=$(sys2.B), AB=$(sys2.AB) after $(alg2.steps) events")

# Under the shared seed both loops draw identical waiting times and channels —
# the generator performs exactly the bookkeeping written out above, so the two
# trajectories coincide event by event.

# ## Trajectory
#
# The system reaches a dynamic equilibrium in which molecules continuously
# associate and dissociate. The sum A + B + 2·AB is conserved throughout.

A_t  = meas2[:A].data
B_t  = meas2[:B].data
AB_t = meas2[:AB].data

plot(measurement_times, A_t;  lw=2, label="A",
     xlabel="Time", ylabel="Count",
     title="Reversible dimerization trajectory",
     size=(700, 280), margin=5Plots.mm)
plot!(measurement_times, B_t;  lw=2, label="B")
plot!(measurement_times, AB_t; lw=2, label="AB")

# ## Equilibrium statistics
#
# Time-averaged counts after the system has equilibrated. The law of mass
# action predicts ``\langle AB \rangle / (\langle A\rangle\langle B\rangle)
# = k_\text{on}/k_\text{off}``.

t_eq  = T / 4   ## discard first quarter as transient
i_eq  = searchsortedfirst(measurement_times, t_eq)

equilibrium_ratio(meas) =
    mean(meas[:AB].data[i_eq:end]) /
    (mean(meas[:A].data[i_eq:end]) * mean(meas[:B].data[i_eq:end]))

println("Mass-action ratio — handwritten: ", round(equilibrium_ratio(meas1); digits=4),
        "  generator: ",                     round(equilibrium_ratio(meas2); digits=4),
        "  exact: ",                         round(sys1.k_on / sys1.k_off;   digits=4))
