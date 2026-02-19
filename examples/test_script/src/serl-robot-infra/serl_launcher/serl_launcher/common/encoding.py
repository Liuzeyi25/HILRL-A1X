from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Handle different dimensionalities properly
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                elif len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
                elif len(state.shape) == 4:
                    # Handle 4D case: (batch, seq_len, 1, state_dim) -> (batch, state_dim)
                    state = state.squeeze(axis=-2)  # Remove the '1' dimension
                    if len(state.shape) == 3:
                        state = rearrange(state, "B T C -> B (T C)")
            
            # Process state through dense layers
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            
            # Ensure both tensors have the same number of dimensions before concatenation
            if len(encoded.shape) != len(state.shape):
                # If encoded is 2D and state is 4D, we need to handle this
                if len(encoded.shape) == 2 and len(state.shape) == 4:
                    # Flatten state to match encoded dimensions
                    state = state.reshape(encoded.shape[0], -1)
                elif len(encoded.shape) == 4 and len(state.shape) == 2:
                    # This case might need different handling based on your specific use case
                    # For now, let's reshape encoded to match state
                    encoded = encoded.reshape(state.shape[0], -1)
            
            encoded = jnp.concatenate([encoded, state], axis=-1)

        # if self.use_proprio:
        #     # project state to embeddings as well
        #     state = observations["state"]
        #     if self.enable_stacking:
        #         # Combine stacking and channels into a single dimension
        #         if len(state.shape) == 2:
        #             state = rearrange(state, "T C -> (T C)")
        #             encoded = encoded.reshape(-1)
        #         if len(state.shape) == 3:
        #             state = rearrange(state, "B T C -> B (T C)")
        #     state = nn.Dense(
        #         self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
        #     )(state)
        #     state = nn.LayerNorm()(state)
        #     state = nn.tanh(state)
        #     encoded = jnp.concatenate([encoded, state], axis=-1)

        return encoded
