# MonteCarloX

[![Dev](https://img.shields.io/badge/docs-stable-blue.svg)](https://zierenberg.github.io/MonteCarloX.jl/dev)
[![Build Status](https://travis-ci.com/zierenberg/MonteCarloX.jl.svg?branch=main)](https://travis-ci.com/zierenberg/MonteCarloX.jl)
[![Codecov](https://codecov.io/gh/zierenberg/MonteCarloX.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/zierenberg/MonteCarloX.jl)

MonteCarloX is an open-source project to implement basic and advanced Monte-Carlo algorithms for general use. The project currently focuses on the programming language [julia](https://julialang.org/), which was specifically developed for scientific use, but future extensions to other languages are intended. 

## Goal
Since Monte-Carlo algorithms are often tailored to specific problems, we here attempt to break them down into small basic functions that can be applied independent of the unterlying models. We thereby separate the algorithmic part from the model part. MonteCarloX will only contain the core algorithmic part and requires additional classes or even packages for models that will build on MonteCarloX. Examples of how these could be designed are collected under `examples`. Different to other simulation packages, the goal of MonteCarloX is **not** to hide the final simulation under simple black-box function calls, but to foster the construction of clean **template simulations** that apply an algorithm where the model can be easily replaced.

## Contribute
MonteCarloX employs continuous integration with unit tests for all functions. This is complicated by the stochastic nature of Monte Carlo simulations, which we include by formulating analytical problems and conducting statistical tests. Right now, we are in the process of developing a stable base that fits the needs of a variety of advanced algorithms but API can still change severely. Once we have a first stable verions, MonteCarloX is intendend as a community project.
