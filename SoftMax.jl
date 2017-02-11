# Solution to Q2b (implement a naive softmax) and Q4a (use max-input-subtraction)
import Base

type SoftMax{T}
    # Buffer of the outputs of fprop, needed during bprop.
    σ::Matrix{T}

    SoftMax() = new(zeros(T, 0, 0))
end

function fprop!{T}(sm::SoftMax{T}, x::Matrix{T})
    # Subtract the largest input for stability (Solution to Q4a).
    x = x .- maximum(x, 2)

    # Compute the standard softmax formula.
    ex = exp.(x)
    return sm.σ = ex ./ sum(ex, 2)
end

function bprop!{T}(sm::SoftMax{T}, ∂y::Matrix{T})
    # Intermediate shared computation, see Q1b.
    sums = [view(∂y, i, :) ⋅ view(sm.σ, i, :) for i=1:size(∂y, 1)]

    # Return the gradient wrt. the input, see Q1b.
    return sm.σ .* (∂y .- sums)
end

params(::SoftMax) = ()
grads(::SoftMax) = ()

Base.show(io::IO, l::SoftMax) = print(io, "SoftMax")
