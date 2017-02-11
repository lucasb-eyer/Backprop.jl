include("LogSoftMax.jl")
include("jacobian.jl")
using Base.Test

X = map(Float32, randn(1, 3))
l = LogSoftMax()

# Test grad. wrt. input X
Dfwd = jacobian_fwd(l, X)
Dbwd = jacobian_bwd(l, X)

@test_approx_eq Dfwd Dbwd
