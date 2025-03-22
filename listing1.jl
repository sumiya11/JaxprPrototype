# 0. Install jax
# using CondaPkg; CondaPkg.add("jax")

# 1. Load the packages
using PythonCall
jax = pyimport("jax")

# 2. Define a simple test function
func_str = """
def f(x):
    return jax.numpy.sin(x) + jax.numpy.cos(x)
"""

# 3. Create namespace and define function
namespace = pydict()
namespace["jax"] = jax
pyexec(func_str, namespace)
py_func = namespace["f"]

# 4. Create test input
x = jax.numpy.array([1,2,3], dtype=jax.numpy.float32)

# 5. Compute a jaxpr
jaxpr = jax.make_jaxpr(py_func)(x)

# 6. Compute a jaxpr of the JVP
jvp_jaxpr = jax._src.interpreters.ad.jvp_jaxpr(jaxpr, [true], true)[0]

# 7. Evaluate the gradient at x
fx, dfx = jax.core.eval_jaxpr(jvp_jaxpr.jaxpr, jvp_jaxpr.consts, x, jax.numpy.ones(3))

# 8. Check
true_fx = jax.lax.sin(x) + jax.lax.cos(x)
true_dfx = jax.lax.cos(x) - jax.lax.sin(x)
@assert Bool((fx == true_fx).all())
@assert Bool((dfx == true_dfx).all())
