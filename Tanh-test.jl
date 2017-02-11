include("Tanh.jl")
include("jacobian.jl")
using Base.Test

X = map(Float32, randn(5, 3))
l = Tanh{Float32}()

# Test grad. wrt. input X
Dfwd = jacobian_fwd(l, X)
Dbwd = jacobian_bwd(l, X)

@test_approx_eq Dfwd Dbwd
