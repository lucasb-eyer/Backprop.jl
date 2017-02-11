# Solution to Q4e (implement LogSoftMax) using expression derived in Q4a-c
import Base

type LogSoftMax{T}
    # Buffer of the outputs of fprop, needed during bprop.
    logσ::Matrix{T}

    LogSoftMax() = new(zeros(T, 0, 0))
end

function fprop!{T}(sm::LogSoftMax{T}, x::Matrix{T})
    # Subtract the largest input for stability (Solution to Q4a).
    x = x .- maximum(x, 2)

    # Compute the simplified log-softmax formula.
    return sm.logσ = x .- log.(sum(exp.(x), 2))
end

function bprop!{T}(sm::LogSoftMax{T}, ∂y::Matrix{T})
    # Return the gradient wrt. the input, see Q4b and Q4c.
    σ = exp.(sm.logσ)
    return ∂y - σ .* sum(∂y, 2)
end

params(::LogSoftMax) = ()
grads(::LogSoftMax) = ()

Base.show(io::IO, l::LogSoftMax) = print(io, "LogSoftMax")
