# Checkpointing: rolling checkpoint/restore via Serialization
#
# Public API
#   ckpt = init_checkpoint(file, linked)       -> create session & write initial checkpoint
#   checkpoint!(ckpt; sweep=...)               -> serialize linked objects + metadata to file
#   state = restore(file)                      -> deserialize checkpoint -> NamedTuple
#   relink!(ckpt, linked)                      -> swap linked objects (e.g. after restore)
#   finalize!(ckpt)                            -> remove checkpoint file

using Serialization

"""
    CheckpointSession

Holds the checkpoint file path and a `NamedTuple` of objects that will be
serialized on every `checkpoint!` call.

The linked objects are serialized together with any keyword metadata
(e.g. `sweep=100`) into a single `NamedTuple` written atomically via a
temporary file + rename.
"""
mutable struct CheckpointSession
    file::String
    linked::NamedTuple
end

"""
    checkpoint!(ckpt::CheckpointSession; kwargs...)

Serialize the linked objects together with keyword metadata to the
checkpoint file.  Uses atomic write (tmp + mv) to avoid corruption.

At least one keyword argument should be provided to track progress
(e.g. `sweep=100`).
"""
function checkpoint!(ckpt::CheckpointSession; kwargs...)
    meta = (; kwargs...)

    for name in keys(ckpt.linked)
        hasproperty(meta, name) &&
            throw(ArgumentError("checkpoint metadata key `$(name)` conflicts with linked object name"))
    end

    tmp = ckpt.file * ".tmp"
    state = (; meta..., ckpt.linked...)
    open(tmp, "w") do io
        serialize(io, state)
    end
    mv(tmp, ckpt.file; force=true)
    return ckpt.file
end

"""
    relink!(ckpt::CheckpointSession, linked::NamedTuple)

Replace the linked objects in a checkpoint session.
Call this after `restore` to attach the newly deserialized objects.
"""
function relink!(ckpt::CheckpointSession, linked::NamedTuple)
    ckpt.linked = linked
    return ckpt
end

"""
    init_checkpoint(file::String, linked::NamedTuple; kwargs...)

Create a `CheckpointSession`, ensure the parent directory exists,
and write an initial checkpoint.  Any keyword arguments (e.g. `sweep=0`)
are stored as metadata in the initial snapshot.

# Example
```julia
ckpt = init_checkpoint("run/ckpt.mcx", (sys=sys, alg=alg); sweep=0)
```
"""
function init_checkpoint(file::String, linked::NamedTuple; kwargs...)
    mkpath(dirname(file))
    ckpt = CheckpointSession(file, linked)
    checkpoint!(ckpt; kwargs...)
    return ckpt
end

"""
    restore(path::String) -> NamedTuple

Deserialize a checkpoint file and return its contents as a `NamedTuple`.
The tuple contains both metadata (e.g. `sweep`) and serialized objects
(e.g. `sys`, `alg`).

# Example
```julia
state = restore("run/ckpt.mcx")
sys = state.sys
alg = state.alg
start = state.sweep + 1
```
"""
function restore(path::String)
    state = open(path, "r") do io
        deserialize(io)
    end
    return state
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
