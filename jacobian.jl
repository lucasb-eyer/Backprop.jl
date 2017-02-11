include("utils.jl")

"""
Computes the jacobian of a layer `l` with respect to `wrt`, evaluated at the
point `x`, using 2nd order finite differences of precision `h`. The resulting
matrix contains the entries of `wrt` along the column and the entries of `l`'s
output along the rows, e.g.:

    J_{i,j} = ∂_i f_j (x)

The flattening of `wrt` and `l`'s output happens in column-major order.
Example:

    x = FloatX[0 1 2]
    l = xavier!(Linear{FloatX}(1,1))
    Jw = jacobian_fwd(l, x, l.w, 1e-5)
    Jx = jacobian_fwd(l, x, x, 1e-5)
"""
function jacobian_fwd(l, x, wrt, h=0)
    out = fprop!(l, x)
    jacobian = Array(eltype(out), (length(out), length(wrt)))

    # Slightly perturb ("wiggle") each parameter individually.
    for (i,idx) in enumerate(eachindex(wrt))
        orig = wrt[idx]

        # See https://codewords.recurse.com/issues/four/hack-the-derivative
        # and the link to lecture notes therein.
        eff_h = h == 0 ? √eps(typeof(orig)) * max(abs(orig), one(orig)) : h

        wrt[idx] = orig + eff_h
        right = vec(fprop!(l, x))

        wrt[idx] = orig - eff_h
        left = vec(fprop!(l, x))

        wrt[idx] = orig
        jacobian[:,i] = (right - left)/2eff_h
    end

    jacobian
end


jacobian_fwd(l, x) = jacobian_fwd(l, x, x)

"""
Uses backprop to compute the jacobian of a layer `l` with respect to `wrt`,
evaluated at the point `x`. The resulting matrix contains the entries of `wrt`
along the column and the entries of `l`'s output along the rows, e.g.:

    J_{i,j} = ∂_i f_j (x)

The flattening of `wrt` and `l`'s output happens in column-major order.
The last parameter, `dwrt` is the buffer of the layer containing the
derivative. This should be automated somehow in the future.

Example:

    x = FloatX[0 1 2]
    l = xavier!(Linear{FloatX}(1,1))
    Jw = jacobian_backward(l, x, l.w, l.dw)
    Jx = jacobian_backward(l, x, x, nothing)
"""
function jacobian_bwd(l, x, wrt, dwrt)
    out = fprop!(l, x)
    jacobian = Array(eltype(out), (length(out), length(wrt)))

    # Set each output to 1, individually, and backprop from it.
    for (i,idx) in enumerate(eachindex(out))
        dout = zeros(out)
        dout[idx] = 1

        zero_grads!(l)

        din = bprop!(l, dout)
        if is(wrt, x)
            jacobian[i,:] = vec(din)
        else
            jacobian[i,:] = vec(dwrt)
        end
    end

    jacobian
end

jacobian_bwd(l, x) = jacobian_bwd(l, x, x, nothing)
