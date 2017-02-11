# Given a container `net`, propagates the input `x` through all layers in `net`.
# Returns the final output of the network.
function forward(net, x)
    for layer in net
        x = fprop!(layer, x)
    end
    return x
end


# Given a container `net`, back-propagates the error-signal ∂y through all layers
# in `net` in reverse order. ∂y is typically a vector filled with 1/N.
# Returns the gradient of the output wrt. the input(s).
function backward(net, ∂y)
    for layer in reverse(net)
        ∂y = bprop!(layer, ∂y)
    end
    return ∂y
end


# Sets all gradients of a given layer to zero.
function zero_grads!(layer)
    for ∂θ in grads(layer)
        ∂θ[:] = 0
    end
end


# Calls `update_fn` on all (parameter, gradient) pairs in the container `net`.
function update_grads!(net, update_fn)
    for layer in net
        map(update_fn, params(layer), grads(layer))
    end
end


# Does a full batch of training: forward, backward, update.
function train_batch(net, crit, xb, yb; λ=0.01)
    target!(crit, yb)
    map(zero_grads!, net)

    # Forward pass
    pred = forward(net, xb)
    costs = fprop!(crit, pred)
    # Or alternatively, the following works just as well:
    # costs = forward([net ; crit], xb)

    # Backward pass
    # ∂pred = bprop!(crit, 1/size(xb, 1))
    ∂pred = bprop!(crit, ones(costs) ./ size(xb, 1))
    ∂x = backward(net, ∂pred)
    # Again, the following works just as well:
    # ∂x = backward([net ; crit], ones(costs) ./ size(xb, 1))

    # Update the parameters using the computed gradients, via SGD.
    update_grads!(net, (θ, ∂θ) -> sgd!(θ, ∂θ, λ))

    return mean(costs)
end


# Updates the parameters `θ` in a single step of SGD using their gradients `∂θ`
# and the learning-rate (update ratio) `λ`.
function sgd!(θ, ∂θ, λ)
    θ[:] = θ - λ*∂θ
end


# Loads train/valid/test split of data used for the exercise.
# Also pre-processes it by converting to float and dividing by 255 to get into [0,1] range.
function loadMNIST(prefix, T::Type)
    Xtr = readdlm("$prefix-train-data.csv", T)
    ytr = readdlm("$prefix-train-probas.csv", T)
    Xva = readdlm("$prefix-valid-data.csv", T)
    yva = readdlm("$prefix-valid-probas.csv", T)
    Xte = readdlm("$prefix-test-data.csv", T)
    yte = readdlm("$prefix-test-probas.csv", T)

    return (Xtr/255-0.5, ytr), (Xva/255-0.5, yva), (Xte/255-0.5, yte)
end
