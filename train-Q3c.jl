include("Linear.jl")
include("Tanh.jl")
include("SoftMax.jl")
include("CrossEntropy.jl")

include("utils.jl")


println("Loading data...")
(Xtr, ttr), (Xva, tva), (Xte, tte) = loadMNIST(length(ARGS) == 1 ? ARGS[1] : "./data/mnist", Float32)

# And define a few training hyperparameters
const epochs = 100
const batch_size = 600
const λ = 0.13
const Nhid = 200

# And a few short-hands for later.
const Ntr, Ndim = size(Xtr)
const Nva = size(Xva, 1)
const Ncls = size(ttr, 2)

# Define the network (just a softmax-regression here)
const net = [
    Linear{Float32}(Ndim, Nhid, σ=1/sqrt(mean([Ndim, Nhid]))),
    Tanh{Float32}(),
    Linear{Float32}(Nhid, Ncls, σ=0),
    SoftMax{Float32}(),
]
const crit = CrossEntropy{Float32}()


# Train for a hundred epochs.
println("Starting training...")
for ep=1:epochs
    # Fit the network.
    cost = 0
    for i=1:batch_size:Ntr
        Xb = Xtr[i:min(i+batch_size-1,Ntr),:]
        tb = ttr[i:min(i+batch_size-1,Ntr),:]
        cost += train_batch(net, crit, Xb, tb, λ=λ)
    end

    # Evaluate on the validation set.
    nerr = 0
    for i=1:batch_size:Nva
        Xb = Xva[i:min(i+batch_size-1,Nva),:]
        tb = tva[i:min(i+batch_size-1,Nva),:]
        # Compute predicted probabilities.
        py = forward(net, Xb)
        # Find the most probable class label.
        _, y = findmax(py, 2)
        _, t = findmax(tb, 2)
        # Count the mistakes.
        nerr += sum(y .!= t)
    end

    @printf("Errors at epoch %d: %d (%.2f%% accuracy, %.2fNLL)\n", ep, nerr, 100*(1 - nerr/Nva), cost/length(1:batch_size:Ntr))
end
