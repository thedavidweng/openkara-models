# ADR-0001: Target ONNX opset 17

## Status

Accepted

## Context

The conversion pipeline exports Demucs PyTorch models to ONNX for cross-platform
inference via official ONNX Runtime builds. The opset version determines which
operators are available and how they accept inputs.

## Decision

Target **opset 17** for all exported ONNX artifacts.

## Consequences

- **ReduceMean** must use the attribute form (`axes=[0]`, `keepdims=0`) rather than
  the input-tensor form. The input-tensor form was introduced in opset 18.
- **Unsqueeze** takes `axes` as an input tensor (initializer), not as an attribute.
  This changed in opset 13; opset 17 inherits the input-tensor form.
- Shape inference must run before `onnx.checker.check_model(full_check=True)` because
  the checker requires type information that shape inference resolves.

These are facts about opset 17, not separate decisions. Any future opset bump must
revisit all three.
