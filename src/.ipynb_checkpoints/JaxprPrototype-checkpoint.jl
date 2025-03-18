module JaxprPrototype

import MacroTools: combinedef

debug() = true

include("MyTinyJaxpr.jl")
include("tracing.jl")
include("vmap.jl")

end # JaxprPrototype
