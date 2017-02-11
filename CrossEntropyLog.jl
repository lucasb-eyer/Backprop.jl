# Solution to Q4e (implement a CrossEntropyLog)
import Base

type CrossEntropyLog{T}
    # Buffer of the targets, needed during fprop and bprop.
    t::Matrix{T}

    CrossEntropyLog() = new(zeros(T, 0, 0))
end

target!{T}(l::CrossEntropyLog{T}, t::Matrix{T}) = l.t = t

function fprop!{T}(l::CrossEntropyLog{T}, x::Matrix{T})
    # No need to remember anything here.
    return -sum(l.t .* x, 2)
end

function bprop!{T}(l::CrossEntropyLog{T}, ∂y::Matrix{T})
    # This expression comes from Q4d. Notice the absence of division,
    # which means there's no potential for instability!
    return ∂y .* -l.t
end

params(::CrossEntropyLog) = ()
grads(::CrossEntropyLog) = ()

Base.show(io::IO, l::CrossEntropyLog) = print(io, "CrossEntropy(Log)")
