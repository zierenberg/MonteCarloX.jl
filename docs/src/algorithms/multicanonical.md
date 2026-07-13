# Multicanonical

The multicanonical (muca) algorithm tabulates a log-weight ``W(x)`` over bins of a reaction coordinate ``x`` so that the chain's marginal distribution in ``x`` is approximately flat across some range.
A flat marginal enables the chain to traverse free-energy barriers that the canonical sampler would never cross in finite time, and lets a single run cover the full range of ``x`` for density-of-states estimation or reweighting to any temperature within that range.

For the shared MCMC protocol — `accept!`, ensembles, the linearity trait — see the [class page](markov_chain_monte_carlo.md).
The closely related Wang-Landau algorithm pursues the same flat-histogram goal with a different weight-refinement strategy and is documented separately.

## Concept

Let ``x`` be a real-valued coordinate (typically an energy, but in general any function of the system state).
If the chain samples a target ``\pi(\text{state}) \propto e^{\text{logweight}(x)}`` and ``\Omega(x)`` is the density of states in ``x``, the marginal is

```math
p(x) \;\propto\; \Omega(x)\, e^{\text{logweight}(x)}.
```

For canonical sampling at inverse temperature ``\beta``, ``\text{logweight}(x) = -\beta x`` and ``p(x)`` is the familiar Boltzmann distribution — typically sharply peaked, leaving the tails and any intermediate-``x`` barriers unsampled.

Multicanonical replaces this with a tabulated ``W(x)`` chosen so that

```math
p_{\text{muca}}(x) \;\propto\; \Omega(x)\, e^{W(x)} \;\approx\; \text{const}.
```

The ideal weight is therefore ``W(x) = -\log \Omega(x)``. The algorithm estimates this weight iteratively by running short sampling phases, accumulating a histogram ``H(x)`` of visits, and using ``H`` to refine ``W``.

## Acceptance rule

The acceptance log-ratio is the difference of tabulated weights at the new and old coordinate values:

```math
\Delta = W(x_{\text{new}}) - W(x_{\text{old}}),
\qquad
\alpha = \min(1,\ e^{\Delta}).
```

`MulticanonicalEnsemble` is non-linear (``W(x)`` is a tabulated function, not affine), so the delta form of `accept!` is invalid.
Always use the full-state form:

```julia
accept!(alg, arg_new, arg_old)
```

The `Multicanonical(rng, bins)` constructor wraps a `MulticanonicalEnsemble` in a generic `MarkovChainMonteCarlo` algorithm and enforces this at construction time.

## Weight refinement

`update_logweight!(ens::MulticanonicalEnsemble; mode)` refines the tabulated weights from the accumulated histogram. Two modes:

### `:simple` — Berg-Neuhaus

```math
W_k \leftarrow W_k - \log H_k.
```

The trivial update: subtract the log-histogram. This corresponds to the original Berg-Neuhaus prescription (Berg & Neuhaus, Phys. Rev. Lett. 68, 9, 1992).
Each iteration's histogram is discarded; only the most recent statistics inform the new weights.

### `:recursive` — precision-weighted

For each adjacent-bin slope ``\Delta_k = W_{k+1} - W_k``, form a new estimate from the current iteration's histogram counts ``H_k, H_{k+1}``:

```math
\Delta_k^{\text{est}} = \log\frac{W_{k+1}}{H_{k+1}} - \log\frac{W_k}{H_k},
\qquad
w_k = \frac{H_k H_{k+1}}{H_k + H_{k+1}}.
```

Update the cumulative slope estimate as a precision-weighted average:

```math
\Delta_k^{\text{new}} = \frac{w_k^{\text{old}}\, \Delta_k^{\text{old}} + w_k\, \Delta_k^{\text{est}}}{w_k^{\text{old}} + w_k}.
```

The cumulative weight ``w_k^{\text{old}}`` grows across iterations (stored in `log_cumweight` in log-space), so converged regions are protected from being disturbed by noise in later iterations.
This is the recursion of Berg (J. Stat. Phys. 82, 323, 1996) and Janke (Physica A 254, 164, 1998); see also Zierenberg, Dissertation, Universität Leipzig (2016), §5.4.2.

**When to use which.** `:simple` is robust during the initial exploration when the weight estimate is far from correct. `:recursive` is more sample-efficient once the chain reaches all bins and produces non-zero counts everywhere — the cumulative-weight bookkeeping protects converged regions from being trampled by short late-stage iterations. A common workflow runs a few `:simple` iterations to find the support, then switches to `:recursive`.

## Usage workflow

The canonical structure has two phases — **weight refinement** (iterate sample → `update!` → reset) and **production** (run with frozen weights for measurements):

```julia
using MonteCarloX, Random

rng  = Xoshiro(1)
bins = BinnedObject(x_min:dx:x_max, 0.0; boundary=NegInfBoundary())
alg  = Multicanonical(rng, bins)

# Phase 1: refine weights
for iter in 1:n_iter
    for _ in 1:n_sample
        # propose a move; compute arg_new and arg_old (the ensemble argument, e.g. reaction coordinate), then:
        accept!(alg, arg_new, arg_old) && commit!(...)
    end
    update_logweight!(ensemble(alg); mode = iter <= 5 ? :simple : :recursive)
    reset!(alg)                     # zero histogram + accept counters
end

# Phase 2: production with frozen weights
ensemble(alg).record_visits = false   # optional: skip histogram bookkeeping
for _ in 1:n_production
    accept!(alg, arg_new, arg_old) && commit!(...)
    measure!(measurements, sys, step)
end
```

The `NegInfBoundary` on the bins means proposals that fall outside the bin range get ``W = -\infty`` and are always rejected — the chain is confined to the explored coordinate range.

### Parallel refinement

`ParallelMulticanonical` runs independent muca chains and merges their histograms between iterations so the weights converge faster.
It is not a separate type — a `ParallelChains` whose algorithms carry a `MulticanonicalEnsemble` qualifies:

```julia
pmuca = ParallelMulticanonical(ThreadsBackend(4), [Multicanonical(Xoshiro(s), bins) for s in 1:4])

for iter in 1:n_iter
    with_parallel(pmuca) do alg
        for _ in 1:n_sample
            accept!(alg, arg_new, arg_old) && commit!(...)
        end
    end
    merge_histograms!(pmuca)                                # accumulate into root
    on_root(pmuca) do i
        update_logweight!(ensemble(algorithm(pmuca, i)); mode=:recursive)
    end
    distribute_logweight!(pmuca)                            # broadcast refined W
    with_parallel(pmuca) do alg; reset!(alg); end
end
```

Both `ThreadsBackend` and `MPIBackend` use the same interface; backend choice is a constructor argument.

### Reweighting to a target ensemble

Once weights are frozen and a sample ``\{x_i\}`` is collected from the muca distribution, canonical expectation values follow by reweighting:

```math
\langle f \rangle_\beta \;=\; \frac{\sum_i f(x_i)\, e^{-\beta x_i - W(x_i)}}{\sum_i e^{-\beta x_i - W(x_i)}}.
```

A single muca run thus produces canonical estimates at *any* ``\beta`` within the explored range, not just one.

## Generic example: a spin gas

A minimal, dependency-free demonstration.
``N`` independent spins with no interaction; the reaction coordinate is the magnetization ``M = \sum_i s_i``. The marginal under uniform sampling is binomial; muca learns ``W(M) \approx -\log \binom{N}{M}`` and produces a flat coverage of ``M \in [0, N]``:

```julia
using MonteCarloX, Random

N     = 64
spins = falses(N)
mag(s) = count(s)

rng  = Xoshiro(42)
bins = BinnedObject(0:1:N, 0.0; boundary=NegInfBoundary())
alg  = Multicanonical(rng, bins)

for iter in 1:30
    for _ in 1:30_000
        i     = rand(rng, 1:N)
        M_old = mag(spins)
        spins[i] = !spins[i]            # propose: flip spin i
        M_new = mag(spins)
        if !accept!(alg, M_new, M_old)
            spins[i] = !spins[i]        # revert
        end
    end
    update_logweight!(ensemble(alg); mode = iter <= 3 ? :simple : :recursive)
    reset!(alg)
end

# After convergence: ensemble(alg).logweight ≈ -log binomial(N, M).
# A subsequent sampling phase visits M ∈ [0, N] approximately uniformly.
```

Everything outside `accept!` and `update!` is system-layer code: picking a site, computing the new coordinate, committing or reverting. The package supplies only the acceptance decision and the weight refinement.

## Full Ising example with `MCXSpins`

For a real system, `MCXSpins.spin_flip!(sys, alg)` dispatches on `alg::AbstractMarkovChainMonteCarlo` to use the full-state form of `accept!`, so the loop body collapses to one call:

```julia
using MonteCarloX, MCXSpins, Random

rng  = Xoshiro(1)
sys  = IsingSystem([32, 32]; J=1)
init!(sys, :hot; rng=rng)

N    = length(sys.spins)
bins = BinnedObject(-2*N:4:2*N, 0.0; boundary=NegInfBoundary())   # 2D Ising: ΔE per flip ∈ {0, ±4, ±8}
alg  = Multicanonical(rng, bins)

for iter in 1:40
    for _ in 1:200_000
        spin_flip!(sys, alg)
    end
    update_logweight!(ensemble(alg); mode = iter <= 5 ? :simple : :recursive)
    reset!(alg)
end
```

The result is a chain that crosses the order/disorder transition freely; a single production run can be reweighted to any temperature spanned by the bins.
For a runnable, plotted version see [Multicanonical sampling: Ising 2D](../generated/muca_Ising2D.md).

## Beyond canonical reweighting: tabulated weights on any coordinate

The name *multicanonical* dates to the original Berg-Neuhaus picture, where the canonical Boltzmann weight ``e^{-\beta E}`` was replaced entirely with ``W(E)``.
The mechanics, however, do not require ``x = E``: ``W`` can tabulate weight over *any* coordinate while the rest of the target stays Boltzmann-distributed, prior-distributed, or governed by some other dynamics.
Two patterns recur:

### Composite ensemble — Blume-Capel with multicanonical crystal field

Wrap multiple ensembles in a user-defined composite type whose `logweight` routes each component to the appropriate sub-ensemble.
For Blume-Capel, the pair interaction ``J \sum_{\langle ij \rangle} s_i s_j`` stays canonical while the crystal-field term ``D \sum_i s_i^2`` is reshaped multicanonically:

```julia
using MonteCarloX

struct CustomEnsemble{B,M} <: AbstractEnsemble
    pair  :: B           # BoltzmannEnsemble for the spin-pair term
    spin2 :: M           # MulticanonicalEnsemble for ∑ s_i²
end

@inline MonteCarloX.logweight(e::CustomEnsemble, H::Tuple{<:Real,<:Real}) =
    MonteCarloX.logweight(e.pair, H[1]) + MonteCarloX.logweight(e.spin2, H[2])

ens = CustomEnsemble(
    BoltzmannEnsemble(β = 0.3),
    MulticanonicalEnsemble(0:1:length(sys.spins)),
)
alg = MetropolisHastingsAlgorithm(rng, ens)

# accept! receives a tuple of sub-energies; only the muca piece is refined later
accept!(alg, (H_pair_new, H_spin2_new), (H_pair_old, H_spin2_old))
update_logweight!(ens.spin2; mode = :recursive)
```

The composite ensemble pattern generalizes: any number of ensembles can be combined, the state passed to `accept!` is a tuple matching the components' inputs, and each component is refined independently.
For the full worked example see [Multicanonical sampling: Blume-Capel](../generated/muca_BlumeCapel.md).

### Derived coordinate — biasing the endpoint of an Ornstein-Uhlenbeck trajectory

Multicanonical can also reshape the distribution of a *derived* coordinate while the underlying dynamics remain governed by their own target.
For trajectories of an Ornstein-Uhlenbeck process under a Gaussian prior, the system maintains the full trajectory and accepts proposed increments under the Gaussian-prior dynamics internally; only the trajectory *endpoint* is submitted to the multicanonical ensemble for re-weighting:

```julia
alg = Multicanonical(rng, BinnedObject(x_min:dx:x_max, 0.0; boundary=NegInfBoundary()))

function step!(sys::OUTrajectory, alg)
    propose_new_increment!(sys)                       # system-defined
    accepted_under_prior(sys) || (revert!(sys); return)
    accept!(alg, sys.xs[end], sys.xs_old[end]) || revert!(sys)
end
```

The outer `accept!` reweights only the endpoint distribution; the rest of the trajectory dynamics is unchanged.
This is the standard route to large-deviation sampling of rare end-points; see [Multicanonical sampling: Ornstein-Uhlenbeck](../generated/muca_OU.md) for the full example.

## Diagnostics & convergence

Useful checks during weight refinement:

- **Histogram flatness.** `flatness(ens.histogram)` returns a scalar in ``[0, 1]`` measuring how uniform the visit counts are across visited bins. Switch from `:simple` to `:recursive` once flatness rises above a threshold (commonly ``0.7`` – ``0.9``), and stop refining when later iterations no longer move it.
- **Roundtrips.** `Roundtrips(x_min, x_max)` counts how often the chain traverses the full coordinate range. Two or more roundtrips per iteration is a stronger convergence signal than flatness alone, especially for first-order-like coexistence regions.
- **Visited range.** `visited_range(ens)` reports the cumulative ``(x_{\min}, x_{\max})`` ever visited. If this range never reaches the bin boundaries, `extend!(ens, :low|:high; anchor, slope)` can paint linear-tail weights past the explored region so the chain doesn't get stuck at a soft cliff.
- **Smoothing.** `smooth!(ens, x_range; window)` applies a moving average to `d_logweight`, useful in late-stage refinement to suppress noise in well-sampled regions.

The combination "flatness > 0.8 AND at least two roundtrips" is a reasonable convergence criterion in practice; pathological landscapes (deep first-order coexistence) may need more.

## API reference

```@docs
MulticanonicalAlgorithm
MulticanonicalEnsemble
update!(e::MulticanonicalEnsemble; mode::Symbol)
visited_range
extend!
smooth!
ParallelMulticanonical
merge_histograms!
distribute_logweight!
flatness
Roundtrips
```
