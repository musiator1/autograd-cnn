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
    p :: Float64
end
dropout(p :: Float64) = Dropout(p)

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
mutable struct GraphNode{OP, N, T}
    args :: NTuple{N, GraphNode}
    grad :: T
    data :: T
end

const GraphWeight = GraphNode{:weight, 0}
const GraphTensor = GraphNode{:tensor, 0}

function GraphNode(data::T, trainable=false) where T
    if trainable
        return GraphNode{:weight, 0, T}((), zero(data), data)
    else
        return GraphNode{:tensor, 0, T}((), zero(data), data)
    end
end

function GraphNode(op::Symbol, args::Tuple, data::T) where T
    N = length(args)
    grad = similar(data)
    return GraphNode{op, N, T}(args, grad, data)
end

show(io::IO, x::GraphNode{OP, N}) where {OP,N} = print(io, "layer ", OP, " with ", N, " arg(s)")
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
    return ordered
end

# Reset gradient for all GraphNodes
function zerograd!(order :: Vector{GraphNode})
    for node in order
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
function forward!(order::Vector{GraphNode}, pairs...)
    for pair in pairs
        tensor, data = pair
        tensor.data .= data
    end

    for node in order
        primal!(node)
    end
end

function backward!(order::Vector{GraphNode})
    seed = last(order)
    seed.grad .= 1

    for node in reverse(order)
        adjoint!(node)
    end
end

function forwardd!(order::Vector{GraphNode}, pairs...)
    for pair in pairs
        tensor, grad = pair
        tensor.grad .= grad
    end

    for node in order
        tangent!(node)
    end
end


# ============================================================================== #
#  5. OPTIMIZERS                                                                 #
# ============================================================================== #
mutable struct GradientDescent
    ∇ :: Dict{GraphWeight, Array{Float64}}
    η :: Float64
    s :: Int64
    GradientDescent(η) = new(Dict(), η, 0)
end

function accumulate!(opt, graph)
    for node in graph
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
    ∇f .= 0.0
    return nothing
end

function optimize!(opt, graph)
    for node in graph
        if node isa GraphWeight
            step!(node.data, opt.η / opt.s, opt.∇[node])
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
    data = zeros(x.outsize...)
    return GraphNode(data)
end

# Losses
function (E::BinaryCrossEntropy)(x, y)
    return GraphNode(:bce, (x, y), zeros(1))
end

function (E::LogitCrossEntropy)(x, y)
    return GraphNode(:lce, (x, y), zeros(1))
end

# Activations
function (y::Sigmoid)(x)
    sz = length(x.data)
    return GraphNode(:sigmoid, (x,), zeros(sz))
end

function (y::ReLU)(x)
    sz  = length(x.data)
    return GraphNode(:relu, (x,), zeros(sz))
end

# Layers
function (y::Dense)(x)
    n   = y.insize
    m   = y.outsize
    limit = sqrt(6 / (n + m))
    W = GraphNode((rand(m, n) .* (2 * limit)) .- limit, true)
    b   = GraphNode(zeros(m), true)
    mul = GraphNode(:mul, (W, x), zeros(m))
    add = GraphNode(:add, (mul, b), zeros(m))
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
    limit = sqrt(6 / (fan_in + fan_out))
    W = GraphNode((rand(fan_in, C_out) .* (2 * limit)) .- limit, true)  # each column is flattened kernel

    # Padding
    H_pad = H_i + 2*p
    W_pad = W_i + 2*p
    x_pad = GraphNode(:pad, (x,), zeros(H_pad, W_pad, C_in))

    # Transform image with im2col
    H_out = H_pad - H_k + 1
    W_out = W_pad - W_k + 1
    x_col = GraphNode(:im2col, (x_pad,), zeros(H_out * W_out, H_k * W_k * C_in))

    # Mul, resize
    mul = GraphNode(:mul, (x_col, W), zeros(H_out * W_out, C_out))
    reshape = GraphNode(:reshape, (mul,), zeros(H_out, W_out, C_out))
    return reshape
end

function (y::MaxPool)(x)
    sz_x = size(x.data)
    sz_out = (div(sz_x[1], y.window_h), div(sz_x[2], y.window_w), sz_x[3])
    return GraphNode(:maxPool, (x,), zeros(sz_out))
end

function (y::Flatten)(x)
    return GraphNode(:reshape, (x,), zeros(length(x.data)))
end

function (d::Dropout)(x)
    sz = size(x.data)
    p_node = GraphNode([d.p])
    mask = zeros(sz)
    mask_node = GraphNode(mask)
    return GraphNode(:dropout, (x, p_node, mask_node), zeros(sz))
end

# OPERATOR PASSES

function primal!(z::GraphNode{:bce, 2})
    function _run(z, x, y)
        z.data = -(y.data .* log.(x.data) + (1 .- y.data) .* log.(1 .- x.data))        
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function adjoint!(z::GraphNode{:bce, 2})
    function _run(z, x, y)
        x.grad -= y.data ./ x.data .* z.grad
        x.grad += (1 .- y.data) ./ (1 .- x.data) .* z.grad     
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function primal!(z::GraphNode{:mul, 2})
    function _run(z, W, x)
        z.data = W.data * x.data
    end
    W, x = z.args
    _run(z, W, x)
    return nothing
end

function adjoint!(z::GraphNode{:mul, 2})
    function _run(z, W, x)
    W.grad += z.grad * x.data'
    x.grad += W.data' * z.grad       
    end
    W, x = z.args
    _run(z, W, x)
    return nothing
end

function primal!(z::GraphNode{:relu, 1})
    function _run(z, x)
        z.data .= max.(0, x.data)
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:relu, 1})
    function _run(z, x)
        for i in 1:length(x.data)
            if x.data[i] == z.data[i]
                x.grad[i] += z.grad[i]
            end
        end
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:add, 2})
    function _run(z, x, y)
        z.data = x.data .+ y.data
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function adjoint!(z::GraphNode{:add, 2})
    function _run(z, x, y)
        x.grad += z.grad
        y.grad += z.grad
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function primal!(z::GraphNode{:dot, 2})
    function _run(z, x, y)
        z.data = dot(x.data, y.data)
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function adjoint!(z::GraphNode{:dot, 2})
    function _run(z, x, y)
        x.grad += y.data .* z.grad
        y.grad += x.data .* z.grad
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function primal!(z::GraphNode{:sum, 1})
    function _run(z, x)
        z.data = sum(x.data)
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:sum, 1})
    function _run(z, x)
        x.grad .+= z.grad
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:sigmoid, 1})
    function _run(z, x)
        z.data = 1 ./ (1 .+ exp.(-x.data))
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:sigmoid, 1})
    function _run(z, x)
        x.grad += exp.(-x.data) ./ (1 .+ exp.(-x.data)) .^ 2 .* z.grad
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:pad, 1})
    function _run(z, x)
        H_out, W_out, _ = size(z.data)
        H_in,  W_in,  _ = size(x.data)
        p_h = div(H_out - H_in, 2)
        p_w = div(W_out - W_in, 2)
        z.data .= 0
        z.data[p_h+1:p_h+H_in, p_w+1:p_w+W_in, :] .= x.data
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:pad, 1})
    function _run(z, x)
        H_out, W_out, _ = size(z.data)
        H_in,  W_in,  _ = size(x.data)
        p_h = div(H_out - H_in, 2)
        p_w = div(W_out - W_in, 2)
        x.grad .+= z.grad[p_h+1:p_h+H_in, p_w+1:p_w+W_in, :]
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:im2col, 1})
    function _run(z, x)
        X = x.data
        H, W, C = size(X)
        N_patches, patch_size = size(z.data)
        HW_k = div(patch_size, C)
        H_k = Int(sqrt(HW_k))
        W_k = H_k
        H_out = H - H_k + 1
        W_out = W - W_k + 1
        idx = 1
        for j in 1:W_out      # column
            for i in 1:H_out  # row
                patch = X[i:i+H_k-1, j:j+W_k-1, :]   # 3d slice 
                z.data[idx, :] .= vec(patch)         # flattened slice into row
                idx += 1
            end
        end
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:im2col, 1})
    function _run(z, x)
        X = x.data
        H, W, C = size(X)
        _, patch_size = size(z.data)
        HW_k = div(patch_size, C)
        H_k = Int(sqrt(HW_k))
        W_k = H_k
        H_out = H - H_k + 1
        W_out = W - W_k + 1
        idx = 1
        for j in 1:W_out
            for i in 1:H_out
                patch_grad = reshape(z.grad[idx, :], (H_k, W_k, C))
                x.grad[i:i+H_k-1, j:j+W_k-1, :] .+= patch_grad
                idx += 1
            end
        end
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:reshape, 1})
    function _run(z, x)
        z.data .= reshape(x.data, size(z.data))
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:reshape, 1})
    function _run(z, x)
        x.grad .+= reshape(z.grad, size(x.data))
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:lce, 2})
    function _run(z, x, y)
        m = maximum(x.data)
        logsumexp = m + log(sum(exp.(x.data .- m)))
        z.data .= -sum(y.data .* x.data) + logsumexp
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function adjoint!(z::GraphNode{:lce, 2})
    function _run(z, x, y)
        ex = exp.(x.data .- maximum(x.data))
        s = ex ./ sum(ex)   # softmax
        x.grad .+= (s .- y.data) .* z.grad
    end
    x, y = z.args
    _run(z, x, y)
    return nothing
end

function primal!(z::GraphNode{:maxPool, 1})
    function _run(z, x)
        H_in, W_in, C = size(x.data)
        H_out, W_out, _ = size(z.data)
        H_w = div(H_in, H_out)
        W_w = div(W_in, W_out)
        for c in 1:C, j in 1:W_out, i in 1:H_out
            h_start = (i-1)*H_w + 1
            w_start = (j-1)*W_w + 1
            z.data[i,j,c] = maximum(
                x.data[h_start:h_start+H_w-1,
                w_start:w_start+W_w-1,
                c]
            )
        end
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function adjoint!(z::GraphNode{:maxPool, 1})
    function _run(z, x)
        H_in, W_in, C = size(x.data)
        H_out, W_out, _ = size(z.grad)
        H_w = div(H_in, H_out)
        W_w = div(W_in, W_out)
        for c in 1:C, j in 1:W_out, i in 1:H_out
            h_start = (i-1)*H_w + 1
            w_start = (j-1)*W_w + 1
            max_val = -Inf
            max_i = 1
            max_j = 1
            window = x.data[h_start:h_start+H_w-1,
                    w_start:w_start+W_w-1,
                    c]
            for wi in 1:H_w, wj in 1:W_w
                v = window[wi, wj]
                if v > max_val
                    max_val = v
                    max_i = wi
                    max_j = wj
                end
            end
            x.grad[h_start+max_i-1,
                w_start+max_j-1,
                c] += z.grad[i,j,c]
        end
    end
    x, = z.args
    _run(z, x)
    return nothing
end

function primal!(z::GraphNode{:dropout, 3})
    function _run(z, x, p_node, mask_node)
        p = p_node.data[1]
        mask_node.data .= rand(size(x.data)) .> p ./ (1 - p)
        z.data .= x.data .* mask_node.data
    end
    x, p_node, mask_node = z.args
    _run(z, x, p_node, mask_node)
    return nothing
end

function adjoint!(z::GraphNode{:dropout, 3})
    function _run(z, x, mask_node)
        x.grad .+= z.grad .* mask_node.data
    end
    x, p_node, mask_node = z.args
    _run(z, x, mask_node)
    return nothing
end
