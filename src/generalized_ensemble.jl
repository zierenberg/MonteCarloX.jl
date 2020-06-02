# generalized ensemble algorithms such as 
# * muca / WL
# * population annealing

#Idea: These structs could actually carry information! E.g the current weight factor for WangLandau (then mutable)
#      in principle also the weights ...
struct Multicanonical end
struct WangLandau end #todo (in terms of log_dos instead of log_weight)


###### discuss: maybe move this to importance sampling (because it is importance sampling)
@doc """
    accept(alg::Multicanonical, rng::AbstractRNG, log_weight::Vector{Float64}, arg_new::T, arg_new::T)::Bool where T

Multicanonical acceptance function
```p(x\\to x^') = \\text{min}\\left(1, W(x^')/W(x)\\right)```
"""
function accept(alg::Multicanonical, rng::AbstractRNG, log_weight::Histogram, arg_new::T, arg_old::T)::Bool where T
    difference = log_weight[args_new] - log_weight[args_old]
    if difference > 0
        return true
    elseif rand(rng) < exp(difference)
        return true
    else
        return false
    end
end


"""

mode can be 
* :simple 
* :extrapolate
* ... ?
"""
function update_weights(alg::Multicanonical, log_weight::Histogram, histogram::Histogram; mode::Symbol=:simple)
end

