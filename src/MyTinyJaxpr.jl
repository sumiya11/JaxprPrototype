
struct Variable
    id::Int
    val::Any
end

unwrap(x::Variable) = x.val
unwrap(x) = x

# We "know" the size and dtype of abstract variables and can dispatch on it
Base.size(x::Variable) = size(unwrap(x))
Base.eltype(x::Variable) = eltype(unwrap(x))

compact_repr_eltype(x) = string(x)
compact_repr_eltype(::Type{Int}) = "i32"
compact_repr_eltype(::Type{Float32}) = "f32"
compact_repr_eltype(::Type{Float64}) = "f64"

compact_repr_size(x::Number) = string(x)
compact_repr_size(x::NTuple) = join(map(compact_repr_size, x), "x")

Base.show(io::IO, x::Variable) = print(io, "x$(x.id)")

struct Equation
    op
    lhs::Variable
    rhs::Vector{Any}
    kwargs
end

input(var::Variable) = Equation(:input, var, [], (;))

Base.show(io::IO, eq::Equation) = print(io, rpad(eq.lhs, 5), "= ", eq.op, " ", rpad(join(map(string, eq.rhs), ", "), 6), isempty(eq.kwargs) ? "" : eq.kwargs)

# See also: https://github.com/jax-ml/jax/blob/55812c5d02d621c9c1c185298efb51ea562da9d6/jax/_src/core.py#L88
struct MyTinyJaxpr
    invars::Vector{Variable}
    outvars::Vector{Variable}
    # Equations are in SSA form
    eqns::Vector{Equation}
end

function to_expr(jaxpr::MyTinyJaxpr)
    exprs = Expr[]
    n_input = length(jaxpr.invars)
    for eq in jaxpr.eqns[n_input+1:end]
        lhs = Symbol(string(eq.lhs))
        rhs = map(x -> x isa Variable ? Symbol(x) : x, eq.rhs)
        op = eq.op
        eq_expr = :($lhs = $op(($(rhs...),)...; $(eq.kwargs)...))
        push!(exprs, eq_expr)
    end
    expr = Expr(:block, exprs...)
    inputs = map(Symbol, jaxpr.invars)
    expr = :(($(inputs...),) -> $expr)
    expr
end

function compile(jaxpr::MyTinyJaxpr)
    expr = to_expr(jaxpr)
    eval(expr)
end

function Base.show(io::IO, jaxpr::MyTinyJaxpr)
    print(io, "MyTinyJaxpr\n")
    print(io, join(map(eq -> rpad("  [" * compact_repr_eltype(eltype(eq.lhs)) * (size(eq.lhs) != () ? " " * compact_repr_size(size(eq.lhs)) : "") * "]", 14) * string(eq), jaxpr.eqns), "\n"))
end
