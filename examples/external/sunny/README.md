# External/Sunny

Examples in this folder demonstrate how MonteCarloX workflows can be connected
with external model packages, starting with Sunny.jl model definitions and
progressing toward MCX-driven algorithm control on external model states.

Each script activates this folder's dedicated environment (`Project.toml`) so
package-compatibility constraints stay isolated from the main examples env.

Current examples:

- `sunny_ising_exact.jl`: Sunny Ising model where updates are driven by MCX
  `MetropolisAlgorithm` acceptance on Sunny proposals/energy deltas, compared
  against Sunny `LocalSampler` and exact 2D Ising energy.
- `sunny_heisenberg.jl`: Sunny classical Heisenberg ferromagnet Monte Carlo
  (magnetization profile over temperature).
- `sunny_cp2_skyrmions.jl`: Sunny SU(3) CP2 skyrmion quench dynamics adapted
  from Sunny example 06 (no plotting dependency).
