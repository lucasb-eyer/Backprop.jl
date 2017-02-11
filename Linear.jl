# Solution to Q2a (implement a Linear module)
import Base

type Linear{T}
    # Parameters of the module
    w::Matrix{T}
    b::Vector{T}

    # (Accumulated) gradients of the parameters and input after backprop
    ∂w::Matrix{T}
    ∂b::Vector{T}
    ∂x::Matrix{T}

    # Buffer of the input to fprop, needed during bprop.
    x::Matrix{T}

    Linear(n_in, n_out ; σ=0) = n_in < 1 || n_out < 1 ? error("Invalid dimensions of Linear") : new(
        map(T, σ*randn(n_in, n_out)),  # w
        zeros(T, n_out),               # b

        zeros(T, n_in, n_out),  # ∂w
        zeros(T, n_out),        # ∂b

        zeros(T, 0, 0),  # ∂x
        zeros(T, 0, 0),  # x
    )
end

function fprop!{T}(l::Linear{T}, x::Matrix{T})
    # Remember the input for later use during bprop.
    l.x = x

    # Compute and return the output.
    return x * l.w .+ l.b'
end

function bprop!{T}(l::Linear{T}, ∂y::Matrix{T})
    # Accumulate the gradients wrt. W and b
    l.∂w[:,:] += l.x' * ∂y
    l.∂b[:] += sum(∂y, 1)'

    # Store and return the gradient wrt. the inputs
    return l.∂x = ∂y * l.w'
end

params(l::Linear) = (l.w, l.b)
grads(l::Linear) = (l.∂w, l.∂b)

Base.show(io::IO, l::Linear) = print(io, "Linear", size(l.w))
