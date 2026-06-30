# Population Monte Carlo

!!! warning "Planned class — not yet implemented"
    Population Monte Carlo is a planned method class. This page records the intended scope so the docs structure reflects the full taxonomy. Nothing here is callable in the current release.

Population Monte Carlo (PMC) methods evolve a *population* of weighted particles ``\{(x_i, w_i)\}_{i=1}^N`` through a sequence of intermediate distributions ``\pi_0, \pi_1, \ldots, \pi_T``. Within each level, particles are mutated by an [MCMC](markov_chain_monte_carlo.md) kernel; between levels, they are reweighted and (optionally) resampled. This sits orthogonally to single-chain MCMC: any MCMC algorithm can serve as the per-particle mutation kernel.

## Why a separate class

The classical answer to "advanced importance sampling in high dimensions" is *not* a better single-sample IS algorithm; it is to build up the target through a sequence of easier intermediate distributions and carry a population through them. The methods below all share that structure:

| Algorithm | Reference | Mechanism |
|---|---|---|
| Annealed Importance Sampling | Neal 2001 | Bridge ``\pi_0 \to \cdots \to \pi_n``; MCMC step at each level; accumulate log-weights |
| SMC samplers | Del Moral, Doucet, Jasra 2006 | AIS + resampling between levels |
| Population Monte Carlo (PMC family) | Cappé et al. 2004; AMIS Cornuet 2012; DM-PMC Elvira et al. 2017 | Iteratively refine proposal from past samples |
| Nested Sampling | Skilling 2004 | Evidence integral via constrained-prior draws |
| Population Annealing | Hukushima & Iba 2003; Machta 2010 | Resample-and-equilibrate sweep across a temperature schedule |
| Cross-entropy method | Rubinstein & Kroese | Outer loop fits parametric ``q`` to high-weight samples (rare events) |

A unified API will be built around four shared ingredients: (1) a sequence or schedule of ensembles, (2) an MCMC mutation step at each level, (3) a weight-accumulation primitive, (4) an optional resampling step. Concrete algorithms will reuse `AbstractEnsemble` and the existing MCMC algorithms as building blocks rather than introducing per-algorithm types.

## Roadmap

- `AnnealedImportanceSampling` — the canonical entry point; composes existing pieces most directly.
- `SMCSampler` — AIS plus resampling on a particle population.
- `PopulationAnnealing` — temperature-schedule resampling, popular in statistical physics.
- `NestedSampling` — evidence computation.
- `PERM` / pruned-enriched Rosenbluth methods — polymer-flavored population MC.

See also: the MCMC [class page](markov_chain_monte_carlo.md) explains why direct and standard importance sampling do not get their own algorithm class — they reduce to one-line user code, and their genuinely scalable variants all land here.
