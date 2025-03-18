# JaxprPrototype

A comprehensive prototype would contain two main parts:
1. Tracing Julia functions to construct a Julia Jaxpr.
2. Communicating with Python to call jax.vmap on Julia Jaxprs.

The present prototype implements only 1.

## Examples

### Basic usage

```julia
using JaxprPrototype

# Declare a method
f(x) = cos(sin(x)^2 + 1)

# Trace at 1.0
JaxprPrototype.trace(f, 1.0)

#= Output
MyTinyJaxpr
  [ f64]      x1   = input 
  [ f64]      x2   = sin x1
  [ f64]      x3   = ^ x2, 2
  [ f64]      x4   = + x3, 1
  [ f64]      x5   = cos x4
=#
```

Control flow is fine, as long as it depends only on concrete values:

```julia
function g(A, v)
   res = []
   for i in 1:3
       push!(res, A^i*v)
   end
   res
end
JaxprPrototype.trace(g, ones(2, 2), ones(2))

#= Output
MyTinyJaxpr
  [2x2 f64]   x1   = input 
  [2 f64]     x2   = input 
  [2x2 f64]   x3   = ^ x1, 1
  [2 f64]     x4   = * x3, x2
  [2x2 f64]   x5   = ^ x1, 2
  [2 f64]     x6   = * x5, x2
  [2x2 f64]   x7   = ^ x1, 3
  [2 f64]     x8   = * x7, x2
=#
```

### Control which arguments are abstracted
  
```julia
function f(x, n)
    n <= 0 && return one(x)
    f(x, n - 1) + f(2x, n - 1)
end

# Would error because `n` is involved in control flow decision 
JaxprPrototype.trace(f, ones(2,2), 2)

# Do not turn `n` into an abstract variable
JaxprPrototype.trace(f, ones(2,2), 1; abstract_arg=[true, false])

#= Output
MyTinyJaxpr
  [2x2 f64]   x1   = input 
  [2x2 f64]   x2   = one x1
  [2x2 f64]   x3   = * 2, x1
  [2x2 f64]   x4   = one x3
  [2x2 f64]   x5   = + x2, x4
=#
```

### Dispatch on size/eltype

```julia
function my_exp(a)
    if isempty(size(a))
        # scalar implementation
        return exp(a)
    else
        # matrix implementation
        @assert length(size(a)) == 2
        return one(a) + a + a^2/2 + a^3/6
    end
end

JaxprPrototype.trace(my_exp, 2.0)
#= Ouput
MyTinyJaxpr
  [ f64]      x1   = input 
  [ f64]      x2   = exp x1
=#

JaxprPrototype.trace(my_exp, ones(2, 2))
#=
MyTinyJaxpr
  [2x2 f64]   x1   = input 
  [2x2 f64]   x2   = one x1
  [2x2 f64]   x3   = ^ x1, 2
  [2x2 f64]   x4   = / x3, 2
  [2x2 f64]   x5   = ^ x1, 3
  [2x2 f64]   x6   = / x5, 6
  [2x2 f64]   x7   = + x2, x1
  [2x2 f64]   x8   = + x7, x4
  [2x2 f64]   x9   = + x8, x6
=#
```

### Put barriers for tracing

```julia
using LinearAlgebra

function f(x, y)
    A1, A2 = ones(2,2), ones(2,2)
    b1, b2 = ones(2), ones(2)
    x = A1*x + b1
    x = A2*x + b2
    LinearAlgebra.norm(x - y)
end

# do not want to descend into `norm` during tracing
JaxprPrototype.trace(f, ones(2), ones(2); do_not_trace=[LinearAlgebra.norm])

#= Output
MyTinyJaxpr
  [2 f64]     x1   = input 
  [2 f64]     x2   = input 
  [2 f64]     x3   = * [1.0 1.0; 1.0 1.0], x1
  [2 f64]     x4   = + x3, [1.0, 1.0]
  [2 f64]     x5   = * [1.0 1.0; 1.0 1.0], x4
  [2 f64]     x6   = + x5, [1.0, 1.0]
  [2 f64]     x7   = - x6, x2
  [ f64]      x8   = norm x7
=#
```
