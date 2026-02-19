import os
import jax
import flax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state, checkpoints
import optax

print("jax   =", jax.__version__)
print("flax  =", flax.__version__)

@struct.dataclass
class SimpleState(train_state.TrainState):
    pass

params = {"w": jnp.ones((4, 4))}
tx = optax.adam(1e-3)
state = SimpleState.create(
    apply_fn=lambda p, x: x @ p["w"],
    params=params,
    tx=tx,
)

# Orbax 要求绝对路径
ckpt_dir = os.path.abspath("./_ckpt_test_orbax")
os.makedirs(ckpt_dir, exist_ok=True)

checkpoints.save_checkpoint(
    ckpt_dir=ckpt_dir,
    target=state,
    step=0,
    overwrite=True,
)

restored = checkpoints.restore_checkpoint(
    ckpt_dir=ckpt_dir,
    target=state,
)

print("restored ok, w[0,0] =", float(restored.params["w"][0, 0]))
