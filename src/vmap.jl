
# Methods vs. functions

import Base.Broadcast: BroadcastFunction, broadcastable

"""
    vmap(jaxpr::MyTinyJaxpr) -> MyTinyJaxpr

Vectorize the function represented by `jaxpr`. 
"""
function vmap(jaxpr::MyTinyJaxpr)
    new_eqs = Vector{Equation}()
    n_input = length(jaxpr.invars)
    append!(new_eqs, jaxpr.eqns[1:n_input])
    varmap = Dict(1:n_input .=> 1:n_input)
    for i in (n_input+1):length(jaxpr.eqns)
        eq = jaxpr.eqns[i]
        for j in 1:length(eq.rhs)
            if eq.rhs[j] isa Variable
                new_rhs_val = unwrap(eq.rhs[j])
                new_eq = Equation(eachslice, Variable(length(new_eqs) + 1, new_rhs_val), [Variable(varmap[eq.rhs[j].id], eq.rhs[j].val)], (dims=1,))
                varmap[eq.rhs[j].id] = length(new_eqs) + 1
            # else
            #     new_rhs_val = Ref(unwrap(eq.rhs[j]))
            #     new_eq = Equation(Ref, Variable(length(new_eqs) + 1, new_rhs_val), [eq.rhs[j]], (;))
                push!(new_eqs, new_eq)
            end
        end
        #
        v_op = BroadcastFunction(eq.op)
        new_lhs_val = unwrap(eq.lhs)
        v_rhs = map(x -> x isa Variable ? Variable(varmap[x.id], x.val) : Ref(x), eq.rhs)
        new_eq = Equation(v_op, Variable(length(new_eqs) + 1, new_lhs_val), v_rhs, (;))
        varmap[eq.lhs.id] = length(new_eqs) + 1
        push!(new_eqs, new_eq)
        # 
        new_eq = Equation(stack, Variable(length(new_eqs) + 1, new_lhs_val), [Variable(length(new_eqs), new_lhs_val)], (dims=1,))
        varmap[eq.lhs.id] = length(new_eqs) + 1
        push!(new_eqs, new_eq)
    end
    MyTinyJaxpr(jaxpr.invars, Variable[], new_eqs)
end
