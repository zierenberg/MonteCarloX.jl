using Random

import Random: rand, SamplerTrivial, CloseOpen12_64, CloseOpen01_64, BitInteger, UInt52Raw, CloseOpen12, SamplerUnion, SamplerType
import Base.reset

"""
    MutableRandomNumbers([rng_base=GLOBAL_RNG], size, mode:=static)

Create a `MutableRandomNumbers` RNG object with a vector of size `size`
containing floating-point random_numbers initially generated with `rng_base`. Random
numbers are then generated sequentially from this vector. The RNG object can be
initialized in two modes:
* :static - then there is an exception thrown once all random numbers are used
* :dynamic - then there are new random numbers generated on the flow from the (copied) rng

# Examples
```julia
julia> rng = MutableRandomNumbers(MersenneTwister(1234),100);

julia> x1 = rand(rng, 2)
2-element Array{Float64,1}:
 0.5908446386657102
 0.7667970365022592

julia> rng = MersenneTwister(1234);

julia> x2 = rand(rng, 2)
2-element Array{Float64,1}:
 0.5908446386657102
 0.7667970365022592

julia> x1 == x2
true
```

Importantly, the random numbers can be accessed and manipulated as in ordinary
array objects.

```jldoctest
julia> rng[1]
0.5908446386657102
julia> rng[3] = 0.2
julia> rand(rng)
0.2
```

Use `reset` in order to rerun the (manipulated) `random` number sequence.

```jldoctest
julia> reset(rng)

```

"""
mutable struct MutableRandomNumbers <: AbstractRNG
    random_numbers::Vector{Float64}
    index_current::Int
    mode::Symbol
    rng_base::AbstractRNG

    function MutableRandomNumbers(rng_base::AbstractRNG, size::Int;
                                  mode::Symbol=:static)
        new(rand(rng_base, size), 0, mode, copy(rng_base))
    end
end
#default when initializing with certain size is a static sequeunce
MutableRandomNumbers(size::Int; mode=:static) = MutableRandomNumbers(Random.GLOBAL_RNG, size, mode=mode)
#default when initializing wihtout size is an emtpy but dynamic sequence
MutableRandomNumbers(; mode=:dynamic) = MutableRandomNumbers(Random.GLOBAL_RNG, 0, mode=mode)

"""
    reset(rng::MutableRandomNumbers, [index::Int=0])

Reset the state of a MutableRandomNumbers object `rng` to `index`. Default
resets the RNG object to the pre-initial index (0)
"""
function reset(r::MutableRandomNumbers, index::Int)
    r.index_current = index
end
reset(r::MutableRandomNumbers) = reset(r,0)

# easy access overloads
Base.length(r::MutableRandomNumbers) = Base.length(r.random_numbers)
Base.getindex(r::MutableRandomNumbers, i::Int) = Base.getindex(r.random_numbers, i)

function Base.setindex!(r::MutableRandomNumbers, value::Real, i::Int)
    if 0 <= value < 1
        Base.setindex!(r.random_numbers, value, i)
    else
        throw(DomainError(value, "random numbers have to be in the domain [0,1)"))
    end
end

# take the next random number from the generator
function gen_rand01(r::MutableRandomNumbers)
    r.index_current += 1
    if checkbounds(Bool, r.random_numbers, r.index_current)
        return @inbounds r.random_numbers[r.index_current]
    else
        if r.mode == :dynamic
            push!(r.random_numbers, rand(r.rng_base))
            return @inbounds r.random_numbers[r.index_current]
        else
            throw(DomainError(r.index_current, "MutableRandomNumbers object has no further random numbers left and is configured in mode `:static`"))
        end
    end
end

#### helper functions

# random numbers work in the range [1,2) because of floating point
# precision, so we have to shift our stored random numbers again
rand_inbounds(r::MutableRandomNumbers, ::CloseOpen12_64) = gen_rand01(r) + 1
rand_inbounds(r::MutableRandomNumbers, ::CloseOpen01_64=CloseOpen01()) =
    rand_inbounds(r, CloseOpen12()) - 1.0

rand_inbounds(r::MutableRandomNumbers, ::UInt52Raw{T}) where {T<:BitInteger} =
    reinterpret(UInt64, rand_inbounds(r, CloseOpen12())) % T

#### generation
function rand(r::MutableRandomNumbers, x::SamplerTrivial{UInt52Raw{UInt64}})
    rand_inbounds(r, x[])
end
rand(r::MutableRandomNumbers, sp::SamplerTrivial{CloseOpen12_64}) = rand_inbounds(r, sp[])

# currently no support for integer cast of random numbers ...
#rand(r::MersenneTwister, T::SamplerUnion(Bool, Int8, UInt8, Int16, Int64, Int128, UInt16, Int32, UInt64, UInt32, UInt128)) =
#    rand(r, UInt52Raw()) % T[]
