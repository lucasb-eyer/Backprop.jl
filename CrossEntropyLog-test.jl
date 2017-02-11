include("CrossEntropyLog.jl")
include("jacobian.jl")
using Base.Test

X = rand(Float32, (5, 3))
t = rand(Float32, (5, 3))
l = CrossEntropyLog{Float32}()

# Test grad. wrt. input X
target!(l, t)
Dfwd = jacobian_fwd(l, X)
Dbwd = jacobian_bwd(l, X)

@test_approx_eq Dfwd Dbwd
