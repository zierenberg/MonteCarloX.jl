# External Integrations

This directory groups examples that connect MonteCarloX workflows to external
package ecosystems.

Planned structure:

- `external/sunny`: Sunny.jl model interoperability and MCX algorithm control.
- `external/smoqy`: SmoQyDQMC.jl determinant QMC driven by MCX replica exchange.
- `external/turing`: Turing.jl / probabilistic-programming interoperability.

Each integration can carry its own isolated environment to avoid dependency
conflicts with the core examples environment.
