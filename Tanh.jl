# Solution to Q3b (implement a tanh module)
import Base

type Tanh{T}
    # Buffer of the output of fprop, needed during bprop.
    out::Matrix{T}

    Tanh() = new(zeros(T, 0, 0))
end

# fprop is trivial
fprop!{T}(l::Tanh{T}, x::Matrix{T}) = l.out = tanh.(x)
# bprop is also relatively simple: build a Jacobian using derivative as in Q3a,
# then left-multipy ∂y with it and this is what you get.
bprop!{T}(l::Tanh{T}, ∂y::Matrix{T}) = ∂y .* (1 .- l.out.^2)

params(l::Tanh) = ()
grads(l::Tanh) = ()

Base.show(io::IO, l::Tanh) = print(io, "Tanh")

