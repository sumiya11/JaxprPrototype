module JaxprPrototype

import MacroTools: combinedef

export trace, compile, to_expr, vmap

debug() = true

include("MyTinyJaxpr.jl")
include("tracing.jl")
include("vmap.jl")

end # JaxprPrototype
