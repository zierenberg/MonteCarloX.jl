#### Random-walk step-size adaptation ####
#
# A warm-up adaptor that tunes the magnitude of a random-walk proposal from the
# accept/reject decisions the Metropolis loop already produces, removing the need to
# hand-tune the step size for a target acceptance rate.

"""
    StepSizeAdaptor(base; target=0.234, rate=1.0, decay=0.7, t0=10)

Warm-up adaptor for a random-walk Metropolis proposal step size.

`base` is the initial step: a scalar for an isotropic proposal, or a per-dimension
vector whose *ratios* are held fixed while the overall magnitude is tuned. During
warm-up, feed each accept/reject decision to [`adapt!`](@ref): the magnitude is
nudged up after an acceptance and down after a rejection, so the running acceptance
rate approaches `target` (`0.234` is near-optimal for random-walk Metropolis; use
`~0.44` in one dimension). Read the current step with [`step_size`](@ref).

The per-step gain is `rate * (t + t0)^{-decay}` (Robbins–Monro):
- `rate` — learning rate; how aggressively the magnitude moves per step.
- `decay ∈ (0.5, 1]` — how fast adaptation cools. `0.7` corrects a several-fold-off
  start within a typical warm-up; `1.0` is the textbook choice but slower to converge
  from far off.
- `t0` — an offset that damps the first few (noisy) updates.

Stop calling `adapt!` once warm-up ends to freeze the step — the sampling phase then
uses a fixed proposal and is exactly reversible.

```julia
step = StepSizeAdaptor(0.1; target=0.234)
for i in 1:(warmup + n)
    θ′  = θ .+ step_size(step) .* randn(rng, length(θ))
    acc = accept!(alg, θ′, θ)
    acc && (θ = θ′)
    i <= warmup && adapt!(step, acc)      # adapt during warm-up only
    ...
end
```
"""
mutable struct StepSizeAdaptor{T}
    base::T
    logscale::Float64
    target::Float64
    rate::Float64
    decay::Float64
    t0::Int
    t::Int
end

StepSizeAdaptor(base; target::Real = 0.234, rate::Real = 1.0, decay::Real = 0.7, t0::Integer = 10) =
    StepSizeAdaptor(base, 0.0, Float64(target), Float64(rate), Float64(decay), Int(t0), 0)

"""
    step_size(a::StepSizeAdaptor)

Current proposal step size: the initial `base` scaled by the adapted magnitude.
"""
@inline step_size(a::StepSizeAdaptor) = a.base .* exp(a.logscale)

"""
    adapt!(a::StepSizeAdaptor, accepted::Bool) -> step

Update the step magnitude from one accept/reject decision (Robbins–Monro) and return
the new [`step_size`](@ref). Call only during warm-up.
"""
@inline function adapt!(a::StepSizeAdaptor, accepted::Bool)
    a.t += 1
    a.logscale += a.rate * (accepted - a.target) / (a.t + a.t0)^a.decay
    return step_size(a)
end

"""
    reset!(a::StepSizeAdaptor)

Reset the adaptation state (magnitude back to `base`, counter to zero).
"""
function reset!(a::StepSizeAdaptor)
    a.logscale = 0.0
    a.t = 0
    return a
end
