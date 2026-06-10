#=
OPTYMALIZACJE:
- dodanie makr @inbounds oraz @simd
=#

# ============================================================================== #
#  1. IMPORTS & BASE ABSTRACTIONS                                                #
# ============================================================================== #
using LinearAlgebra
import Base: show
abstract type Operator end
abstract type Loss end

# ============================================================================== #
#  2. OPERATORS & LAYERS (Activations, Dense, Conv, etc.)                        #
# ============================================================================== #

# Operators
struct Sigmoid <: Operator end
sigmoid() = Sigmoid()

struct ReLU <: Operator end
relu() = ReLU()

struct Dense <: Operator
    insize  :: Int64
    outsize :: Int64
end
dense(pair :: Pair{Int64, Int64}, activation) = tuple(dense(pair), activation())
dense(pair :: Pair{Int64, Int64}) = Dense(first(pair), last(pair))

struct Conv <: Operator
    kernel_size  :: Tuple{Int64, Int64}
    in_channels  :: Int64
    out_channels :: Int64
    pad_size     :: Int64
end
conv(kernel_size :: Tuple{Int64, Int64}, pair :: Pair{Int64, Int64}, pad_size :: Int64) = 
    Conv(kernel_size, first(pair), last(pair), pad_size)

struct MaxPool <: Operator
    window_h :: Int64
    window_w :: Int64
end
maxPool(win_size :: Tuple{Int64, Int64}) = MaxPool(win_size...)

struct Flatten <: Operator end
flatten() = Flatten()

struct Dropout <: Operator 
    p :: Float32
end
dropout(p :: Float32) = Dropout(p)

# Loss functions
struct BinaryCrossEntropy <: Loss end
bce(output, target) = BinaryCrossEntropy()(output, target)

struct LogitCrossEntropy  <: Loss end
lce(output, target) = LogitCrossEntropy()(output, target)

# Define tensor
struct Tensor{N}
    outsize :: NTuple{N, Int64}
end

tensor(sz...) = Tensor(sz)()

# Define chain with pseudo-constructor and function for Chain() instances
const Chain = Vector{Operator}

function chain(operators::Tuple)
    y = Vector{Operator}()
    for v in operators
        if v isa Tuple
            push!(y, v...)
        else
            push!(y, v)
        end
    end
    return y
end

function (chain::Chain)(x)
    node = x
    for op in chain
        node = op(node)
    end
    return node
end

# ============================================================================== #
#  3. GRAPH DATA STRUCTURES (Tensor & GraphNode)                                 #
# ============================================================================== #
# Define GraphNode and constructors
mutable struct GraphNode{OP, A <: Tuple, A, D, G}
    args :: A
    grad :: G
    data :: D
end

const GraphWeight = GraphNode{:weight}
const GraphTensor = GraphNode{:tensor}

function GraphNode(data::D, trainable=false) where D
    grad = zero(data)

    if trainable
        return GraphNode{:weight, Tuple{}, D, typeof(grad)}((), grad, data)
    else
        return GraphNode{:tensor, Tuple{}, D, typeof(grad)}((), grad, data)
    end
end

function GraphNode(op::Symbol, args::A, data::D) where {A <: Tuple, D}
    grad = similar(data)
    return GraphNode{op, A, D, typeof(grad)}(args, grad, data)
end

show(io::IO, x::GraphNode{OP}) where {OP} = print(io, "layer ", OP)
show(io::IO, x::GraphWeight) = print(io, "weight")
show(io::IO, x::GraphTensor) = print(io, "tensor")

# ============================================================================== #
#  4. AUTOGRAD ENGINE (Forward & Backward API)                                   #
# ============================================================================== #
# Main functions controlling the flow: graph(), zerograd!(), forward!(), backward!()
# Make ordered list from computional graph
function graph(node)
    function visit!(node::GraphNode, visited, ordered)
        if node in visited
        else
            push!(visited, node)
            for arg in node.args
                visit!(arg, visited, ordered)
            end
            push!(ordered, node)
        end
        return nothing
    end

    ordered = Vector{GraphNode}()
    visited = Set{GraphNode}()
    visit!(node, visited, ordered)
    return Tuple(ordered)
end

# Reset gradient for all GraphNodes
function zerograd!(order::Tuple)
    foreach(order) do node
        node.grad .= 0
    end
end

# Functions for API compatibility
function primal!(tensor::GraphTensor) end
function primal!(weight::GraphWeight) end
function tangent!(tensor::GraphTensor) end
function tangent!(weight::GraphWeight) end
function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end

# Forward pass, forward automatic differentiation, backward automatic differentiation
function forward!(order::Tuple, pairs...)
    for pair in pairs
        pair[1].data .= pair[2]
    end
    foreach(primal!, order) 
end

function backward!(order::Tuple)
    seed = last(order)
    seed.grad .= 1f0
    foreach(adjoint!, reverse(order))
end

function forwardd!(order::Tuple, pairs...)
    for pair in pairs
        tensor, grad = pair
        tensor.grad .= grad
    end
    foreach(tangent!, order)
end

# ============================================================================== #
#  5. OPTIMIZERS                                                                 #
# ============================================================================== #
mutable struct GradientDescent
    ∇ :: Dict{GraphWeight, Array{Float32}}
    η :: Float32
    s :: Int64
    GradientDescent(η) = new(Dict(), η, 0)
end

function accumulate!(opt, order::Tuple)
    foreach(order) do node
        if node isa GraphWeight
            if node in keys(opt.∇)
                opt.∇[node] .+= node.grad
            else
                opt.∇[node] = copy(node.grad)
            end
        end
    end
    opt.s += 1
    return nothing
end

function step!(θ, α, ∇f)
    θ .-= α .* ∇f
    ∇f .= 0f0
    return nothing
end

function optimize!(opt, order::Tuple)
    foreach(order) do node
        if node isa GraphWeight
            # Type assertion (::typeof) zapobiega czerwonemu kolorowi przy wyciąganiu ze słownika Any
            grad_acc = opt.∇[node]::typeof(node.grad)
            step!(node.data, opt.η / opt.s, grad_acc)
        end
    end
    opt.s = 0
    return nothing
end

# ============================================================================== #
#  6. PROPAGATION RULES (Primal & Adjoint Passes)                                #
# ============================================================================== #

# OPERATOR DISCRETIZATION

# Tensor
function (x::Tensor{N})() where N
    data = zeros(Float32, x.outsize...)
    return GraphNode(data)
end

# Losses
function (E::BinaryCrossEntropy)(x, y)
    return GraphNode(:bce, (x, y), zeros(Float32, 1))
end

function (E::LogitCrossEntropy)(x, y)
    return GraphNode(:lce, (x, y), zeros(Float32, 1))
end

# Activations
function (y::Sigmoid)(x)
    sz = length(x.data)
    return GraphNode(:sigmoid, (x,), zeros(Float32, sz))
end

function (y::ReLU)(x)
    sz  = length(x.data)
    return GraphNode(:relu, (x,), zeros(Float32, sz))
end

# Layers
function (y::Dense)(x)
    n   = y.insize
    m   = y.outsize
    limit = Float32(sqrt(6 / (n + m)))
    W = GraphNode((rand(Float32, m, n) .* (2f0 * limit)) .- limit, true)
    b   = GraphNode(zeros(Float32, m), true)
    mul = GraphNode(:mul, (W, x), zeros(Float32, m))
    add = GraphNode(:add, (mul, b), zeros(Float32, m))
    return add
end

function (y::Conv)(x)
    # Get kernel and image size
    H_k, W_k = y.kernel_size
    C_in = y.in_channels
    C_out = y.out_channels
    p = y.pad_size
    H_i, W_i, C_i = size(x.data)

    # Glorot uniform
    fan_in  = H_k * W_k * C_in
    fan_out = H_k * W_k * C_out
    limit = Float32(sqrt(6 / (fan_in + fan_out)))
    W = GraphNode((rand(Float32, fan_in, C_out) .* (2f0 * limit)) .- limit, true)  # each column is flattened kernel

    # Padding
    H_pad = H_i + 2*p
    W_pad = W_i + 2*p
    x_pad = GraphNode(:pad, (x,), zeros(Float32, H_pad, W_pad, C_in))

    # Transform image with im2col
    H_out = H_pad - H_k + 1
    W_out = W_pad - W_k + 1
    patch_size = H_k * W_k * C_in
    N_patches = H_out * W_out
    x_col = GraphNode(:im2col, (x_pad,), zeros(Float32, patch_size, N_patches))

    # Mul, resize
    mul = GraphNode(:mul_At_B, (x_col, W), zeros(Float32, N_patches, C_out))
    reshape = GraphNode(:reshape, (mul,), zeros(Float32, H_out, W_out, C_out))
    return reshape
end

function (y::MaxPool)(x)
    sz_x = size(x.data)
    sz_out = (div(sz_x[1], y.window_h), div(sz_x[2], y.window_w), sz_x[3])
    return GraphNode(:maxPool, (x,), zeros(Float32, sz_out))
end

function (y::Flatten)(x)
    return GraphNode(:reshape, (x,), zeros(Float32, length(x.data)))
end

function (d::Dropout)(x)
    sz = size(x.data)
    p_node = GraphNode([d.p])
    mask = zeros(Float32, sz)
    mask_node = GraphNode(mask)
    return GraphNode(:dropout, (x, p_node, mask_node), zeros(Float32, sz))
end

# OPERATOR PASSES

function primal!(z::GraphNode{:bce})
    x, y = z.args
    z.data .= -(y.data .* log.(x.data) + (1 .- y.data) .* log.(1 .- x.data))        
    return nothing
end

function adjoint!(z::GraphNode{:bce})
    x, y = z.args
    x.grad .-= y.data ./ x.data .* z.grad
    x.grad .+= (1 .- y.data) ./ (1 .- x.data) .* z.grad     
    return nothing
end

function primal!(z::GraphNode{:mul})
    W, x = z.args
    mul!(z.data, W.data, x.data)
    return nothing
end

function adjoint!(z::GraphNode{:mul})
    W, x = z.args
    mul!(W.grad, z.grad, x.data', 1.0f0, 1.0f0)
    mul!(x.grad, W.data', z.grad, 1.0f0, 1.0f0)       
    return nothing
end

# NEW
function primal!(z::GraphNode{:mul_At_B})
    x, W = z.args
    mul!(z.data, x.data', W.data)
    return nothing
end

# NEW
function adjoint!(z::GraphNode{:mul_At_B})
    x, W = z.args
    mul!(x.grad, W.data, z.grad', 1.0f0, 1.0f0)
    mul!(W.grad, x.data, z.grad, 1.0f0, 1.0f0)       
    return nothing
end

function primal!(z::GraphNode{:relu})
    x, = z.args
    z.data .= max.(0f0, x.data)
    return nothing
end

function adjoint!(z::GraphNode{:relu})
    x, = z.args
    @inbounds @simd for i in 1:length(x.data)
        if x.data[i] == z.data[i]
            x.grad[i] += z.grad[i]
        end
    end
    return nothing
end

function primal!(z::GraphNode{:add})
    x, y = z.args
    z.data .= x.data .+ y.data
    return nothing
end

function adjoint!(z::GraphNode{:add})
    x, y = z.args
    x.grad .+= z.grad
    y.grad .+= z.grad
    return nothing
end

function primal!(z::GraphNode{:dot})
    x, y = z.args
    z.data .= dot(x.data, y.data)
    return nothing
end

function adjoint!(z::GraphNode{:dot})
    x, y = z.args
    x.grad .+= y.data .* z.grad
    y.grad .+= x.data .* z.grad
    return nothing
end

function primal!(z::GraphNode{:sum})
    x, = z.args
    z.data .= sum(x.data)
    return nothing
end

function adjoint!(z::GraphNode{:sum})
    x, = z.args
    x.grad .+= z.grad
    return nothing
end

function primal!(z::GraphNode{:sigmoid})
    x, = z.args
    z.data .= 1 ./ (1 .+ exp.(-x.data))
    return nothing
end

function adjoint!(z::GraphNode{:sigmoid})
    x, = z.args
    x.grad .+= exp.(-x.data) ./ (1 .+ exp.(-x.data)) .^ 2 .* z.grad
    return nothing
end

function primal!(z::GraphNode{:pad})
    x, = z.args
    H_out, W_out, _ = size(z.data)
    H_in,  W_in,  _ = size(x.data)
    p_h = div(H_out - H_in, 2)
    p_w = div(W_out - W_in, 2)
    z.data .= 0f0
    z.data[p_h+1:p_h+H_in, p_w+1:p_w+W_in, :] .= x.data
    return nothing
end

function adjoint!(z::GraphNode{:pad})
    x, = z.args
    H_out, W_out, _ = size(z.data)
    H_in,  W_in,  _ = size(x.data)
    p_h = div(H_out - H_in, 2)
    p_w = div(W_out - W_in, 2)
    x.grad .+= z.grad[p_h+1:p_h+H_in, p_w+1:p_w+W_in, :]
    return nothing
end

function primal!(z::GraphNode{:im2col})
    x, = z.args
    X = x.data
    H, W, C = size(X)
    patch_size, _ = size(z.data)
    HW_k = div(patch_size, C)
    H_k = Int(sqrt(HW_k))
    W_k = H_k
    H_out = H - H_k + 1
    W_out = W - W_k + 1

    idx = 1
    for j in 1:W_out, i in 1:H_out
        patch_idx = 1
        for c in 1:C, w in 1:W_k
            @inbounds @simd for h in 1:H_k
                z.data[patch_idx, idx] = X[i+h-1, j+w-1, c]
                patch_idx += 1
            end
        end
        idx += 1
    end
    return nothing
end

function adjoint!(z::GraphNode{:im2col})
    x, = z.args
    X = x.data
    H, W, C = size(X)
    patch_size, _ = size(z.data)
    HW_k = div(patch_size, C)
    H_k = Int(sqrt(HW_k))
    W_k = H_k
    H_out = H - H_k + 1
    W_out = W - W_k + 1

    idx = 1
    for j in 1:W_out, i in 1:H_out
        patch_idx = 1
        for c in 1:C, w in 1:W_k
            @inbounds @simd for h in 1:H_k
                x.grad[i+h-1, j+w-1, c] += z.grad[patch_idx, idx]
                patch_idx += 1
            end
        end
        idx += 1
    end
    return nothing
end

function primal!(z::GraphNode{:reshape})
    x, = z.args
    z.data .= reshape(x.data, size(z.data))
    return nothing
end

function adjoint!(z::GraphNode{:reshape})
    x, = z.args
    x.grad .+= reshape(z.grad, size(x.data))
    return nothing
end

function primal!(z::GraphNode{:lce})
    x, y = z.args
    m = maximum(x.data)
    logsumexp = m + log(sum(exp.(x.data .- m)))
    z.data .= -sum(y.data .* x.data) + logsumexp
    return nothing
end

function adjoint!(z::GraphNode{:lce})
    x, y = z.args
    ex = exp.(x.data .- maximum(x.data))
    s = ex ./ sum(ex)   # softmax
    x.grad .+= (s .- y.data) .* z.grad
    return nothing
end

function primal!(z::GraphNode{:maxPool})
    x, = z.args
    H_in, W_in, C = size(x.data)
    H_out, W_out, _ = size(z.data)
    H_w = div(H_in, H_out)
    W_w = div(W_in, W_out)
    
    for c in 1:C, j in 1:W_out, i in 1:H_out
        h_start = (i-1)*H_w
        w_start = (j-1)*W_w

        max_val = -Inf32
        for wj in 1:W_w
            @inbounds @simd for wi in 1:H_w
                value = x.data[h_start + wi, w_start + wj, c]
                if value > max_val
                    max_val = value
                end
            end
        end
        @inbounds z.data[i, j, c] = max_val
    end
    return nothing
end

function adjoint!(z::GraphNode{:maxPool})
    x, = z.args
    H_in, W_in, C = size(x.data)
    H_out, W_out, _ = size(z.grad)
    H_w = div(H_in, H_out)
    W_w = div(W_in, W_out)
    
    for c in 1:C, j in 1:W_out, i in 1:H_out
        h_start = (i-1)*H_w
        w_start = (j-1)*W_w

        max_val = -Inf32
        max_h = 1
        max_w = 1

        @inbounds for wj in 1:W_w, wi in 1:H_w 
            value = x.data[h_start + wi, w_start + wj, c]
            if value > max_val
                max_val = value
                max_h = h_start + wi
                max_w = w_start + wj
            end
        end
        @inbounds x.grad[max_h, max_w, c] += z.grad[i, j, c]
    end
    return nothing
end

function primal!(z::GraphNode{:dropout})
    x, p_node, mask_node = z.args
    p = p_node.data[1]
    mask_node.data .= rand(Float32, size(x.data)) .> p ./ (1 - p)
    z.data .= x.data .* mask_node.data
    return nothing
end

function adjoint!(z::GraphNode{:dropout})
    x, p_node, mask_node = z.args
    x.grad .+= z.grad .* mask_node.data
    return nothing
end