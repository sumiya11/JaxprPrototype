using PythonCall

#=
Setting up the environment
using CondaPkg; CondaPkg.add("jax")
=#

jax = pyimport("jax")
# numpy = pyimport("numpy")

# Define a simple test function
func_str = """
def f(x):
    return jax.numpy.sin(x) + jax.numpy.cos(x)

def f3(x):
    return jax.numpy.sin(x) * jax.numpy.cos(x)

def f2(x):
    return x*x
"""

# Create namespace and define function
# https://discourse.julialang.org/t/calling-python-function-jit-compiled-with-jax-from-julia-without-overhead/123552
namespace = pydict()
namespace["jax"] = jax
pyexec(func_str, namespace)
py_func = namespace["f"]

# Create test input
x = jax.numpy.array([1,2,3], dtype=jax.numpy.float32)
py_func(2.0)
py_func(x)

###
traced = jax.jit(py_func)
lowered = traced.lower(x)
compiled = lowered.compile()
compiled(x)

###
jaxpr2 = jax.make_jaxpr(namespace["f2"])(x)
jax_mul_primitive = jaxpr2.eqns[0].primitive

###
jaxpr = jax.make_jaxpr(py_func)(x)

###
jvp_jaxpr = jax._src.interpreters.ad.jvp_jaxpr(jaxpr, [true], true)[0]

###
jaxpr_new = jax.make_jaxpr(py_func)(x)
eq = jaxpr_new.eqns[2]
new_eq = jax.core.new_jaxpr_eqn(eq.invars, eq.outvars, jax_mul_primitive, eq.params, eq.effects)

###
jaxpr_new.eqns[2] = new_eq
jax._src.interpreters.ad.jvp_jaxpr(jaxpr_new, [true], true)[0]


