# Measurement framework
# Based on examples/api.ipynb

using StatsBase

"""
    Measurement{F,T}

A measurement consisting of an observable function and a data container.

# Fields
- `observable::F`: Function that extracts a value from the system
- `data::T`: Container for storing measurements (Vector, Histogram, etc.)
"""
struct Measurement{F,T}
    observable::F
    data::T
end

"""
    Measurement(pair::Pair)

Create a Measurement from a Pair of observable function => data container.
"""
Measurement(pair::Pair) = Measurement(pair.first, pair.second)

"""
    measure!(measurement::Measurement, sys; kwargs...)

Perform a single measurement by evaluating the observable on the system
and storing the result.
"""
function measure!(measurement::Measurement, sys; kwargs...)
    value = measurement.observable(sys)
    push!(measurement.data, value)
end

"""
    reset!(measurement::Measurement)

Reset a single measurement to its initial state by clearing its data container.
For histogram-backed measurements this zeros the bin counts while preserving binning.
"""
function reset!(measurement::Measurement)
    _reset_data!(measurement.data)
    return measurement
end

@inline _reset_data!(data::AbstractVector) = (empty!(data); data)

function _reset_data!(data::Histogram)
    fill!(data.weights, zero(eltype(data.weights)))
    return data
end

function _reset_data!(data)
    if hasmethod(empty!, Tuple{typeof(data)})
        empty!(data)
        return data
    end
    throw(ArgumentError("unsupported measurement data container $(typeof(data)); define `reset!` support for this container"))
end

# Measurement schedules

"""
    MeasurementSchedule

Abstract type for different measurement scheduling strategies.
"""
abstract type MeasurementSchedule end

"""
    IntervalSchedule <: MeasurementSchedule

Schedule measurements at regular intervals.

# Fields
- `interval::Float64`: Time/step interval between measurements
- `_checkpoint::Float64`: Internal checkpoint for next measurement
"""
mutable struct IntervalSchedule <: MeasurementSchedule
    interval::Float64
    _checkpoint::Float64
    IntervalSchedule(interval) = new(interval, 0.0)
end

"""
    PreallocatedSchedule <: MeasurementSchedule

Schedule measurements at pre-specified times.

# Fields
- `times::Vector{Float64}`: Sorted vector of measurement times
- `checkpoint_idx::Int`: Current index in times vector
"""
mutable struct PreallocatedSchedule <: MeasurementSchedule
    times::Vector{Float64}
    checkpoint_idx::Int
    PreallocatedSchedule(times) = new(sort(times), 1)
end

"""
    reset!(schedule::MeasurementSchedule)

Reset schedule counters/checkpoints back to their initial state.
"""
function reset!(schedule::IntervalSchedule)
    schedule._checkpoint = 0.0
    return schedule
end

function reset!(schedule::PreallocatedSchedule)
    schedule.checkpoint_idx = 1
    return schedule
end

# Main measurements container

"""
    Measurements{K,S<:MeasurementSchedule}

Container for multiple measurements with a shared schedule.

# Fields
- `measurements::Dict{K, Measurement}`: Dictionary of named measurements
- `schedule::S`: Measurement schedule
"""
mutable struct Measurements{K,S<:MeasurementSchedule}
    measurements::Dict{K, Measurement}
    schedule::S
end

"""
    times(m::Measurements)

Return the measurement time points for preallocated schedules.
"""
times(m::Measurements{K, PreallocatedSchedule}) where K = m.schedule.times

"""
    data(m::Measurements{K}, key::K)

Return the raw data container for a named measurement.
"""
data(m::Measurements{K}, key::K) where K = m[key].data

measurement_data(m::Measurements{K}, key::K) where K = data(m, key)

"""
    Measurements(measurements::Dict{K, Measurement}; interval::Real)

Create Measurements with an interval-based schedule.
"""
function Measurements(measurements::Dict{K, Measurement}; interval::Real) where K
    schedule = IntervalSchedule(interval)
    Measurements{K, typeof(schedule)}(measurements, schedule)
end

"""
    Measurements(measurements::Dict{K, Measurement}, times::Vector{<:Real})

Create Measurements with a preallocated time-based schedule.
"""
function Measurements(measurements::Dict{K, Measurement}, times::Vector{<:Real}) where K
    schedule = PreallocatedSchedule(Float64.(times))
    Measurements{K, typeof(schedule)}(measurements, schedule)
end

"""
    Measurements(pairs::Vector{<:Pair{K}}; interval::Real)

Create Measurements from a vector of name => (observable => data) pairs
with an interval-based schedule.
"""
function Measurements(pairs::Vector{<:Pair{K}}; interval::Real) where K <: Union{Symbol, String}
    measurements = Dict{K, Measurement}(name => Measurement(pair) for (name, pair) in pairs)
    Measurements(measurements, interval=interval)
end

"""
    Measurements(pairs::Vector{<:Pair{K}}, times::Vector{<:Real})

Create Measurements from a vector of name => (observable => data) pairs
with a preallocated time-based schedule.
"""
function Measurements(pairs::Vector{<:Pair{K}}, times::Vector{<:Real}) where K <: Union{Symbol, String}
    measurements = Dict{K, Measurement}(name => Measurement(pair) for (name, pair) in pairs)
    Measurements(measurements, times)
end

# Indexing interface
Base.getindex(m::Measurements{K}, key::K) where K = m.measurements[key]
Base.setindex!(m::Measurements{K}, val, key::K) where K = m.measurements[key] = val

"""
    measure!(measurements::Measurements{K, IntervalSchedule}, sys, t; kwargs...)

Perform measurements at regular intervals (indefinite simulation).
"""
function measure!(measurements::Measurements{K, IntervalSchedule}, sys, t; kwargs...) where K
    schedule = measurements.schedule
    if t >= schedule._checkpoint
        for (name, measurement) in measurements.measurements
            measure!(measurement, sys; kwargs...)
        end
        schedule._checkpoint += schedule.interval
    end
end

"""
    measure!(measurements::Measurements{K, PreallocatedSchedule}, sys, t; kwargs...)

Perform measurements at preallocated times (handles event skipping).
"""
function measure!(measurements::Measurements{K, PreallocatedSchedule}, sys, t; kwargs...) where K
    schedule = measurements.schedule
    # Process all checkpoints that have been passed
    while schedule.checkpoint_idx <= length(schedule.times) &&
        t >= schedule.times[schedule.checkpoint_idx]
            
        for (name, measurement) in measurements.measurements
            measure!(measurement, sys; kwargs...)
        end
        
        schedule.checkpoint_idx += 1
    end
end

"""
    reset!(measurements::Measurements)

Reset all measurement data containers and schedule state in-place.
"""
function reset!(measurements::Measurements)
    for measurement in values(measurements.measurements)
        reset!(measurement)
    end
    reset!(measurements.schedule)
    return measurements
end

"""
    is_complete(m::Measurements)

Check if all scheduled measurements are complete.
Returns false for interval-based schedules (indefinite).
"""
is_complete(m::Measurements{K, IntervalSchedule}) where K = false  # indefinite
is_complete(m::Measurements{K, PreallocatedSchedule}) where K = 
    m.schedule.checkpoint_idx > length(m.schedule.times)
