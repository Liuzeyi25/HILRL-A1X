from typing import Any, Mapping, Sequence, Union

import jax

# Handle different JAX versions
try:
    PRNGKey = jax.random.KeyArray
except AttributeError:
    # For older JAX versions or newer versions where KeyArray is removed
    PRNGKey = jax.Array
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
