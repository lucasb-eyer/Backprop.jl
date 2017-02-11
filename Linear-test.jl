include("Linear.jl")
include("jacobian.jl")
using Base.Test

X = map(Float32, randn(5, 3))
l = Linear{Float32}(3, 4)

# Test grad. wrt. parameter W
Dfwd = jacobian_fwd(l, X, l.w)
Dbwd = jacobian_bwd(l, X, l.w, l.∂w)

@test_approx_eq Dfwd Dbwd

# Test grad. wrt. parameter b
Dfwd = jacobian_fwd(l, X, l.b)
Dbwd = jacobian_bwd(l, X, l.b, l.∂b)

@test_approx_eq Dfwd Dbwd

# Test grad. wrt. input X
Dfwd = jacobian_fwd(l, X)
Dbwd = jacobian_bwd(l, X)

@test_approx_eq Dfwd Dbwd
