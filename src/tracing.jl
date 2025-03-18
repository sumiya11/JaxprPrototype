# Global for convenience.
TAPE = Ref([])

# Returns a "traced" function  that wraps `f`.
# The traced function can be evaluated at `args`.
function make_traced_function(f, args)
    argtypes = map(typeof, args)
    args_unwrapped = map(unwrap, args)
    @assert !any(arg -> arg isa Variable, args_unwrapped)
    jaxpr = @__MODULE__
    name = nameof(f)
    argnames = map(i -> gensym(), 1:length(args))
    sig_args = [:($name::$type) for (name, type) in zip(argnames, argtypes)]
    kwargs, whereparams = [], ()
    body = quote
        $jaxpr.debug() && printstyled("debug: ", $name, " ", ($(argnames...),), "\n", color=:red)
        # Invoke the method on concrete values
        res = $name(($args_unwrapped)...)
        # Record to tape
        lhs = $jaxpr.Variable(length($jaxpr.TAPE[])+1, res)
        rhs = ($(argnames...),)
        equation = $jaxpr.Equation($f, lhs, collect(rhs))
        push!($jaxpr.TAPE[], equation)
        lhs
    end
    g = combinedef(Dict(:name => name, :args => sig_args, :kwargs => kwargs, :body => body, :whereparams => whereparams)) 
    g
end

"""
    trace(f, args...) -> MyTinyJaxpr

Constructs a `MyTinyJaxpr` of the evaluation of `f` at `args`.

## Optional arguments

- `abstract_arg`: an array of `Bool`s. 
    Whether to make the arguments into abstract values. 
    By default, `True` for all arguments.

- `do_not_trace`: an array of functions. 
    Do not descend into those functions during tracing.
"""
function trace(
        f, args...; 
        abstract_arg=trues(length(args)),
        do_not_trace=[]
    )
    # check that f runs fine by itself
    f(args...)
    # wrap arguments
    args = map(i -> abstract_arg[i] ? Variable(i, args[i]) : args[i], eachindex(args))
    # create barriers for tracer
    for h in do_not_trace
        # this is not correct usually; here only for demonstration
        g = make_traced_function(h, args[1:1])
        m = parentmodule(h)
        m.eval(g)
    end
    iters = 0
    # to avoid an infinite loop in unsupported cases
    while iters < 1000
        iters += 1
        TAPE[] = map(arg -> input(arg), args[abstract_arg])
        try
            Base.invokelatest(f, args...)
            break
        catch e
            # There is no method for e.f(..., ::Variable, ...).
            # Construct a traced method and put it into global scope.
            if e isa MethodError
                g = make_traced_function(e.f, e.args)
                m = parentmodule(e.f)
                m.eval(g)
            else
                rethrow(e)
            end
        end
    end
    MyTinyJaxpr(collect(args[abstract_arg]), Variable[], TAPE[])
end
