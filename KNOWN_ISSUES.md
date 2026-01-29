# Known Issues and Workarounds

This document tracks known issues in vLLM-Omni and their workarounds.

## vLLM Dependency Conflict (v0.14.0)

**Status**: Open
**Priority**: High
**Affects**: v0.14.0rc1

### Issue
The vLLM dependency (`vllm==0.14.0`) is currently commented out in `pyproject.toml` (line 47) due to an entrypoints overwrite problem.

```python
# "vllm==0.14.0",  # TODO: fix the entrypoints overwrite problem
```

### Impact
- Users may need to manually install vLLM separately
- Potential compatibility issues if wrong vLLM version is used
- Unclear dependency requirements for new users

### Workaround
Until this is resolved, users should:
1. Install vLLM manually if needed:
   ```bash
   pip install vllm==0.14.0
   ```
2. Be aware of potential entrypoint conflicts

### Root Cause
Both vLLM and vLLM-Omni define entrypoints that conflict with each other, likely in the `[project.scripts]` section of their respective `pyproject.toml` files.

### Potential Solutions
1. **Rename Entrypoints**: Rename vLLM-Omni entrypoints to avoid conflicts (e.g., `vllm-omni` instead of `vllm`)
2. **Conditional Installation**: Make vLLM an optional dependency
3. **Coordinate with vLLM**: Work with vLLM team to resolve the conflict upstream
4. **Entry Point Merging**: Implement a mechanism to merge or conditionally register entry points

### Next Steps
- [ ] Investigate exact entrypoint conflicts
- [ ] Propose solution to vLLM-Omni team
- [ ] Coordinate with vLLM upstream if needed
- [ ] Update documentation to clarify relationship with vLLM

## Other Known Issues

### Large Python Files
Several Python files exceed 1000 lines, which may impact maintainability:
- `vllm_omni/model_executor/models/qwen3_tts/modeling_qwen3_tts.py` (2307 lines)
- `vllm_omni/entrypoints/openai/serving_chat.py` (2142 lines)
- `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_token2wav.py` (1877 lines)

**Recommendation**: Consider refactoring these into smaller, more focused modules.

### TODO Items in Code
Multiple TODO items exist in the codebase (30+ instances). Priority items include:
- Auto version generation in `vllm_omni/version.py`
- Multimodal config validation in `vllm_omni/engine/arg_utils.py`
- Logging configuration in `vllm_omni/entrypoints/omni_diffusion.py`
- Request abortion for stages in async operations

## Reporting New Issues

If you discover a new issue:
1. Check this document and GitHub issues to avoid duplicates
2. For security issues, follow [SECURITY.md](SECURITY.md)
3. For bugs and features, open a GitHub issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU, etc.)
