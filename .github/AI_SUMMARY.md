# MonteCarloX for AI Agents

MonteCarloX is a Julia framework for Monte Carlo and stochastic simulation. It keeps algorithms separate from models and favors small, composable building blocks.

Mission and design
- Separation of concerns: algorithms live in MonteCarloX, models live elsewhere (e.g., SpinSystems).
- No systems in core: MonteCarloX contains algorithms only, not model implementations.
- Composability: the same algorithm works with any compatible system.

Core ideas
- Algorithms are lightweight structs with explicit state and counters.
- Logweight functions define target distributions and are passed into algorithms.
- Measurements are explicit and scheduled; data flows through a small measurement API.

Core abstractions
- AbstractSystem
- AbstractAlgorithm
- AbstractLogWeight
- AbstractUpdate
- AbstractMeasurement

Key modules
- src/algorithms: Metropolis and other samplers.
- src/weights: canonical logweights for equilibrium sampling.
- src/measurements: Measurement and scheduling utilities.
- src/utils: histogram helpers and KL divergence.

Measurement schedules
- IntervalSchedule: measure every N steps.
- PreallocatedSchedule: measure at specific times.

Typical workflow
- Define a logweight (target distribution).
- Construct an algorithm (e.g., Metropolis) with RNG and logweight.
- Propose updates, compute log ratios, call accept!.
- Measure observables with Measurement or Measurements.
- Validate distributions with histograms or KL divergence.

Tests
- Tests mirror core modules in test/.
- Metropolis tests cover 1D and 2D Gaussians, acceptance tracking, temperature effects, and proposal invariance.

Notes for AI agents
- Prefer local closures in tests and examples to avoid global state.
- Keep logweight coupled to the algorithm instance.
- Use Measurements to collect time series and histograms when comparing distributions.
