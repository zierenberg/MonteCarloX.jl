# Checkpointing: rolling checkpoint/restore via Serialization
#
# Public API
#   ckpt = init_checkpoint(file, state; kw...)   -> create session & write initial checkpoint
#   checkpoint!(ckpt; kw...)                     -> serialize state (merged with kw) to file
#   ckpt = restore_checkpoint(file)              -> deserialize -> new CheckpointSession
#   finalize!(ckpt)                              -> remove checkpoint file
#   ckpt.sys, ckpt.alg, ...                      -> access stored objects directly

using Serialization

"""
    CheckpointSession

Holds the checkpoint file path and a `NamedTuple` of objects that will be
serialized on every `checkpoint!` call.

Fields are accessed directly on the session object:

```julia
ckpt = restore_checkpoint("run/ckpt.mcx")
ckpt.sys          # restored system
ckpt.alg          # restored algorithm
ckpt.sweep        # metadata counter
ckpt.file         # checkpoint file path (reserved)
```

Keyword arguments passed to `checkpoint!` override same-named fields,
allowing progress counters (e.g. `sweep`) to be updated without mutating
the stored tuple.
"""
mutable struct CheckpointSession
    file::String
    _state::NamedTuple
end

function Base.getproperty(ckpt::CheckpointSession, name::Symbol)
    name === :file || name === :_state ? getfield(ckpt, name) : getfield(ckpt, :_state)[name]
end

"""
    checkpoint!(ckpt::CheckpointSession; kwargs...)

Serialize the stored objects to the checkpoint file.  Keyword arguments
override same-named fields (e.g. `sweep=100` updates the stored sweep
counter).  Uses atomic write (tmp + mv) to avoid corruption.
"""
function checkpoint!(ckpt::CheckpointSession; kwargs...)
    state = merge(getfield(ckpt, :_state), (; kwargs...))
    tmp = ckpt.file * ".tmp"
    open(tmp, "w") do io
        serialize(io, state)
    end
    mv(tmp, ckpt.file; force=true)
    return ckpt.file
end

"""
    init_checkpoint(file::String, state::NamedTuple; kwargs...)

Create a `CheckpointSession`, ensure the parent directory exists,
and write an initial checkpoint.  Any keyword arguments (e.g. `sweep=0`)
override same-named fields in `state` for the initial snapshot.

# Example
```julia
ckpt = init_checkpoint("run/ckpt.mcx", (sys=sys, alg=alg, sweep=0))
```
"""
function init_checkpoint(file::String, state::NamedTuple; kwargs...)
    mkpath(dirname(file))
    ckpt = CheckpointSession(file, state)
    checkpoint!(ckpt; kwargs...)
    return ckpt
end

"""
    restore_checkpoint(path::String) -> CheckpointSession

Deserialize a checkpoint file and return a new `CheckpointSession`.
Access restored fields directly on the returned object.

# Example
```julia
ckpt  = restore_checkpoint("run/ckpt.mcx")
sys   = ckpt.sys
alg   = ckpt.alg
start = ckpt.sweep + 1
```
"""
function restore_checkpoint(path::String)
    state = open(path, "r") do io
        deserialize(io)
    end
    return CheckpointSession(path, state)
end

"""
    finalize!(ckpt::CheckpointSession)

Remove the checkpoint file.  Call this after a simulation completes
successfully so the checkpoint does not linger.
"""
function finalize!(ckpt::CheckpointSession)
    rm(ckpt.file; force=true)
    return nothing
end
