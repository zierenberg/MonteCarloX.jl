```@meta
DocTestSetup = quote
    using MyModule
end
```

# MonteCarloX.jl

*MonteCarloX.jl* is an open-source julia framework for basic and advanced Monte-Carlo simulations of equilibrium and non-equilibrium problems. The basci functionality is developed with physics problems in mind but is intended to generalize beyond, e.g., for chemical reactions etc.

## Goal
Since Monte-Carlo algorithms are often tailored to specific problems, we here attempt to break them down into small basic functions that can be applied independent of the unterlying models. We thereby separate the algorithmic part from the model part. MonteCarloX will only contain the core algorithmic part and requires additional classes or even packages for models that will build on MonteCarloX. Examples of how these could be designed are collected under `examples`. Different to other simulation packages, the goal of MonteCarloX is **not** to hide the final simulation under simple black-box function calls, but to foster the construction of clean **template simulations** that apply an algorithm where the model can be easily replaced.

## Contribute
MonteCarloX employs continuous integration with unit tests for all functions. This is complicated by the stochastic nature of Monte Carlo simulations, which we include by formulating analytical problems and conducting statistical tests. Right now, we are in the process of developing a stable base that fits the needs of a variety of advanced algorithms but API can still change severely. Once we have a first stable verions, MonteCarloX is intendend as a community project.

