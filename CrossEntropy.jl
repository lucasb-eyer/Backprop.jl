# Solution to Q2c (Implement a CrossEntropy)
import Base

type CrossEntropy{T}
    # Just keep these in memory for later inspection, not really necessary.
    costs::Matrix{T}

    # Buffer of the inputs and targets to fprop, needed during bprop.
    x::Matrix{T}
    t::Matrix{T}

    CrossEntropy() = new(
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
    )
end

target!{T}(l::CrossEntropy{T}, t::Matrix{T}) = l.t = t

function fprop!{T}(l::CrossEntropy{T}, x::Matrix{T})
    l.x = x

    # This is the plain formula of cross-entropy, no special trick here.
    return l.costs = - sum(l.t .* log.(x), 2)
end

function bprop!{T}(l::CrossEntropy{T}, ∂y::Matrix{T})
    # This comes from Q1c.
    # Note that ∂y is usually a vector of 1/N where N is the mini-batch size,
    # this way we average over the minibatch and get stable performance wrt. N.
    # It could also be used to give different weight to different data!
    return ∂y .* (- l.t ./ l.x)
end

params(::CrossEntropy) = ()
grads(::CrossEntropy) = ()

Base.show(io::IO, l::CrossEntropy) = print(io, "CrossEntropy")
