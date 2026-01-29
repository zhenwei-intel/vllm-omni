# Changelog

All notable changes to vLLM-Omni will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SECURITY.md file with vulnerability reporting guidelines
- CHANGELOG.md to track version changes

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.14.0rc1] - 2026-01-XX

### Added
- Major RC milestone focused on maturing the diffusion stack
- Strengthened OpenAI-compatible serving
- Expanded omni-model coverage
- Improved stability across platforms (GPU/NPU/ROCm)
- Flash Attention 3 support (fa3-fwd==0.0.1)
- OpenAI Whisper integration (>=20250625)
- ONNX Runtime support (>=1.19.0)
- SOX audio processing support (>=1.5.0)

### Changed
- Improved diffusion transformer architecture
- Enhanced distributed inference with OmniConnectors
- Better KV cache management
- Upgraded accelerate to 1.12.0
- Upgraded Gradio to 5.50

### Known Issues
- vLLM dependency (0.14.0) is currently commented out due to entrypoints overwrite problem
- Several TODO items in codebase related to optimization and refactoring

## [0.12.0rc1] - 2025-11-XX

### Added
- Initial release of vllm-project/vllm-omni
- Omni-modality support (text, image, video, audio)
- Non-autoregressive architectures (Diffusion Transformers)
- Heterogeneous output support
- Multi-platform support (CUDA, ROCm, XPU, NPU)
- OpenAI-compatible API server
- Tensor, pipeline, data and expert parallelism
- Streaming outputs
- Support for Qwen-Omni and Qwen-Image models

### Documentation
- Comprehensive documentation on ReadTheDocs
- Installation and quickstart guides
- Model support documentation
- Contributing guidelines

---

## Version History Notes

### Release Versioning
- **0.14.x**: Release candidate series focusing on stability and feature maturity
- **0.12.x**: Initial public release series

### Breaking Changes
Future breaking changes will be clearly documented here with migration guides.

### Upgrade Guide
Detailed upgrade instructions will be provided for major version transitions.

[Unreleased]: https://github.com/vllm-project/vllm-omni/compare/v0.14.0rc1...HEAD
[0.14.0rc1]: https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0rc1
[0.12.0rc1]: https://github.com/vllm-project/vllm-omni/releases/tag/v0.12.0rc1
