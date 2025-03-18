# - Too many methods created
# - Global scope modified
# - O(methods) evaluations of f
# - Careful treatment of constants and output variables
# - Whether to create a list of functions that we do not want to trace or to implement them ourselves (like jax.lax) 

module JaxprPrototype

import MacroTools: combinedef

debug() = true

include("MyTinyJaxpr.jl")
include("tracing.jl")
include("vmap.jl")

end # JaxprPrototype
