# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test torch compile for ZImageTransformerBlock."""

import pytest
import torch

from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformerBlock


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("modulation", [True, False])
def test_zimage_transformer_block_compile(dtype: torch.dtype, modulation: bool):
    """Test that ZImageTransformerBlock produces the same output with and without torch.compile."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    layer_id = 0
    dim = 512
    n_heads = 8
    n_kv_heads = 8
    norm_eps = 1e-5
    qk_norm = True
    
    # Create the transformer block
    block = ZImageTransformerBlock(
        layer_id=layer_id,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        norm_eps=norm_eps,
        qk_norm=qk_norm,
        modulation=modulation,
    )
    
    # Move to appropriate device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = block.to(device).to(dtype)
    block.eval()
    
    # Create test input tensors
    batch_size = 2
    seq_len = 16
    head_dim = dim // n_heads
    
    # Input hidden states
    x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    
    # Attention mask (all ones for simplicity)
    attn_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)
    
    # RoPE embeddings (cos and sin)
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=dtype, device=device)
    
    # AdaLN input (only needed if modulation is True)
    adaln_input = None
    if modulation:
        adaln_embed_dim = min(dim, 256)  # ADALN_EMBED_DIM from the module
        adaln_input = torch.randn(batch_size, adaln_embed_dim, dtype=dtype, device=device)
    
    # Run without compile
    with torch.no_grad():
        output_no_compile = block(
            x.clone(),
            attn_mask,
            cos,
            sin,
            adaln_input=adaln_input.clone() if adaln_input is not None else None,
        )
    
    # Compile the block
    compiled_block = torch.compile(block)
    
    # Run with compile
    with torch.no_grad():
        output_compiled = compiled_block(
            x.clone(),
            attn_mask,
            cos,
            sin,
            adaln_input=adaln_input.clone() if adaln_input is not None else None,
        )
    
    # Compare outputs
    assert output_no_compile.shape == output_compiled.shape, (
        f"Output shapes don't match: {output_no_compile.shape} vs {output_compiled.shape}"
    )
    
    # Check that outputs are close
    # Use appropriate tolerance based on dtype
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    elif dtype == torch.float16:
        atol, rtol = 1e-3, 1e-2
    else:  # bfloat16
        atol, rtol = 1e-2, 1e-2
    
    max_diff = torch.abs(output_no_compile - output_compiled).max().item()
    mean_diff = torch.abs(output_no_compile - output_compiled).mean().item()
    
    # Calculate relative difference
    output_abs = torch.abs(output_no_compile)
    relative_diff = torch.abs(output_no_compile - output_compiled) / (output_abs + 1e-8)
    max_relative_diff = relative_diff.max().item()
    
    assert torch.allclose(output_no_compile, output_compiled, atol=atol, rtol=rtol), (
        f"Outputs don't match for dtype={dtype}, modulation={modulation}:\n"
        f"  Max absolute difference: {max_diff:.6e}\n"
        f"  Mean absolute difference: {mean_diff:.6e}\n"
        f"  Max relative difference: {max_relative_diff:.6e}\n"
        f"  Tolerance: atol={atol}, rtol={rtol}\n"
        f"  Output range (no compile): [{output_no_compile.min().item():.6e}, {output_no_compile.max().item():.6e}]\n"
        f"  Output range (compiled): [{output_compiled.min().item():.6e}, {output_compiled.max().item():.6e}]"
    )
    
    print(f"✓ Test passed for dtype={dtype}, modulation={modulation}")
    print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")


if __name__ == "__main__":
    # Run tests directly for quick validation
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for modulation in [True, False]:
            print(f"\nTesting dtype={dtype}, modulation={modulation}")
            test_zimage_transformer_block_compile(dtype, modulation)
    print("\n✓ All tests passed!")
