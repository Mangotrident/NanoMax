# Changelog

All notable changes to the AND-CP project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-29

### Added
- **Interactive Control Room**: Implemented Streamlit dashboard featuring multi-channel oscilloscope, Welch-based spectral power and SNR calculations, and live prediction history.
- **REST and WebSocket APIs**: Created FastAPI server including WebSocket streaming, health status diagnostics, batch predictions, and config managers.
- **Deep Learning Model Export**: Integrated ONNX and TorchScript serialization in PyTorch pipelines (`EEGNet`, `ShallowConvNet`, `DeepConvNet`).
- **Pathological Simulator**: Real-time signal degradation mimicking Cerebral Palsy noise contours.
- **Stratified Few-Shot Calibration**: Dynamic patient-specific adaptation using stratified sampling to eliminate class imbalance issues during adapter updates.
- **Continuous Integration**: Configured GitHub Actions workflows for automated code style validations and verification tests.
- **Open-source documentation templates**: Standardized `CONTRIBUTING.md`, `LICENSE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `CHANGELOG.md`.
