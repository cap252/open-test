# Quantization logic

This guide follows `quantize()` as it converts an FP32 model to INT8 QDQ. Paths are relative to `src/opennpu_quant/`.

```text
feeds → calibrate → CalibrationStats → select_ranges → quantize_model → QuantizationResult → (apply_adaround)
        two passes  min/max+histograms MinMax/Pct/Ent    Q/DQ·INT8·INT32   model + audit         relearn codes only
```

## 1. Input requirements

`quantize()` in `quant/api.py` validates the following before processing:

- Graph inputs must be FP32 features or INT64/BOOL structural inputs such as indices and masks.
- Supported ONNX opsets are 13–21.
- `OrtConfig.optimization` must be `ORT_DISABLE_ALL` to prevent ORT from fusing or removing Q/DQ.
- Models that already contain Q/DQ are rejected (`engine.infer_model`).

## 2. Selecting quantization targets: `engine.layout()`

An operator-role table at the top of `engine.py` defines which inputs are features (activations).

| Category | Example operators | Inputs to quantize |
|---|---|---|
| `FIRST` | Conv, Gemm, Pool, Normalization, activations such as Relu/Sigmoid/Gelu, Reshape/Transpose/Resize, DepthToSpace/SpaceToDepth, Softmax | First input only. Other inputs are parameters such as weights, shapes or axes |
| `ALL_INPUTS` | Add, Sub, Mul, Div, Pow, Min, Max, MatMul, Concat, comparisons | All inputs |
| `Where` | Where | Inputs 1 and 2, excluding condition input 0 |
| `STRUCTURAL` | Shape, Size, Constant, NonZero, Range, And/Or/Not | None. Outputs are also excluded |
| Other | ConvTranspose, Einsum, TopK, etc. | Fails with "Explicit input-role policy required". The entire graph is checked even for `basic` scope |

- Scope:
  - `basic` selects inputs to Conv, Gemm and MatMul.
  - `all` selects all feature inputs in the table.
- Tensors derived from constants, including initializers, Constant nodes and their Reshape/Transpose operations, are not activations.
- Graph outputs also receive Q/DQ. DQ inherits the public output name, so classification logits lie on the INT8 quantization grid.
- **Protected boundaries:** `recognize()` in `graph/activations.py` identifies decomposed activations such as `x * sigmoid(x)`, tanh-approximate GELU, decomposed Clip and softmax. Q/DQ is inserted at their boundaries, not internal edges.
  - Ambiguous or overlapping patterns are rejected, such as an `a * sigmoid(b)` gate without supporting Split/Slice provenance.

`layout()` returns a plan containing:

- `tensors`: tensors to quantize.
- `edges`: consumer inputs that must read DQ outputs.
- `rows`: per-node records saved in `quantization.json`.
- `protection`: protected boundaries.

## 3. Calibration: `quant/calibration.py`

`calibrate(model, feeds)` replays `feeds()` **twice**. The argument must be a zero-argument function returning a new iterator each time.

1. Expose all planned tensors as graph outputs and create an ORT session. Planning always uses `all` scope, allowing the same statistics to serve both `basic` and `all`.
2. **First pass:** record each tensor's minimum and maximum.
3. **Second pass:** use the first pass's maximum absolute value `absmax` to fix the interval `±(absmax × 1.00001 + 1e-6)`. Accumulate two histograms with `histogram_bins` bins (default 2048):
   - Signed values.
   - Absolute values `|x|`.
4. Inputs must match in both order and value between passes. Per-sample digests are compared and mismatches fail. This is why a list or an already-created generator is rejected.

`CalibrationStats` contains min/max values, histograms, protected boundaries, the ORT environment and their identities. Stored statistics are revalidated against the model, histogram settings, ORT environment and content hashes when loaded (`quant/storage.py`).

## 4. Range selection: `quant/ranges.py`

| Method | Computation |
|---|---|
| `MinMax` | Use the first pass's min/max directly |
| `Percentile` | Supply collected histograms to ORT `HistogramCollector`. Symmetric selection uses the `\|x\|` histogram, and asymmetric selection uses both tails of the signed histogram |
| `Entropy` | ORT entropy (KL) search using the signed histogram |

ORT clips the result to the observed [min, max]. Histograms are supplied through ORT internals, making the process version-sensitive. The validated ORT version is pinned to 1.20.2.

## 5. Activation scales and zero points

Activations use per-tensor signed INT8. ORT `compute_scale_zp` first computes the base scale (`engine.activation_params`).

```text
Symmetric:  code range [-127, 127],  scale = absmax / 127,       zero_point = 0
Asymmetric: include 0 in the range, scale = (max - min) / 255,  zero_point = round(-128 - min / scale)
Zero range: scale = 1, zero_point = 0
```

PoT policies adjust the base scale to a power of two (`quant/scales.py`).

```text
pot_nearest:  scale = 2^rint(log2 s)    (round in exponent space, exact ties use the even exponent)
pot_ceil:     scale = 2^ceil(log2 s)    (at least the base scale, preserving range coverage)
```

- “Nearest” refers to exponent-space distance. The boundary is `√2 · 2^k`, so 0.72 (log2 = −0.47) becomes 1.0 even though 0.5 is closer in linear distance.
- For asymmetric quantization, recompute the zero point using the PoT-adjusted scale: `clip(rint(-128 - min(low, 0) / scale), -128, 127)`. Range selection itself is not reoptimized for PoT.
- Scales must be within the FP32 normal range, with exponents −126 through 127.

## 6. Weights and biases: `engine._quantize_weights_and_biases`

Targets are Conv, Gemm and MatMul with initializer weights.

- **Weights:** symmetric signed INT8 with zero point 0.
  - Per-tensor or per-channel. The per-channel axis is the output channel: axis 0 for Conv and transposed (`transB`) Gemm, otherwise the last axis.
  - `float`: ORT `quantize_data` computes `scale = absmax / 127`.
  - `pot_nearest` / `pot_ceil`: adjust each channel's base scale to PoT.
  - `pot_mse`: choose between the lower and upper PoT candidates using the reconstruction MSE of the actual stored INT8 codes (`scales.choose_weight_scale`).
- **Biases:** INT32 with per-channel `bias_scale = activation_scale × weight_scale` and zero point 0.
  - Increase the weight scale if bias values exceed INT32 limits. Float policies use ORT's bias-scale adjustment. PoT policies round up to the next power of two at or above the required minimum scale.
- MatMul with non-initializer constant weights, such as Constant or Transpose(initializer), remains FP32. This is a known limitation.

## 7. Graph reconstruction and verification

Reconstruct the graph with input Q/DQ and weight/bias DQ before each applicable node, followed by output Q/DQ. Remove the original FP32 initializers, then perform these checks:

- ONNX checker with full checking.
- `verify()`: planned edges read DQ, protected internal edges have no Q/DQ, and parameters match the policy.
- `parameters.verify_parameters`: code/scale/zero-point shapes, exact PoT scales (mantissa 0.5), and bias scale = activation scale × weight scale.
- Detailed scale audit (`audit["scale_audit"]`): records and checks activation Q/DQ connections, scales, zero points, weight axes, INT8 codes, INT32 biases and scale products for both float and PoT policies. It does not convert float scales to PoT.
- General runs additionally call `audit_runtime` to verify that required Q/DQ remains in ORT's executed graph.

These records are returned in `QuantizationResult.audit` and saved as `quantization.json` by `run`.

## 8. AdaRound: `quant/adaround.py`

AdaRound keeps RTN scales, zero points, biases and graph connections fixed, changing **only INT8 weight codes**.

- **Layer order:** optimize one layer at a time in graph order. Each layer uses:
  - Inputs from the QDQ student, including previously optimized layers.
  - Target outputs from the FP32 teacher.
- **Variables:** each element has `floor(w / s)` and a trainable `alpha`.
  - Soft codes are `floor + h(alpha)`, where `h = clip(1.2·sigmoid(alpha) − 0.1, 0, 1)`. This rectified sigmoid uses ζ = 1.1 and γ = −0.1.
  - After training, hard codes are `floor + (alpha ≥ 0)`.
  - Initialize `alpha` from the fractional part. Integer elements and those saturated at −128/127 are fixed.
- **Loss:** squared error between the Torch layer implementation (`adaround_ops.operation`) and teacher output, summed over channels and averaged over the batch.
  - After warm-up (default 20%), add `λ · Σ(1 − |2h − 1|^β)`. β follows a cosine schedule from 20 to 2.
- **Training:** Adam with learning rate 1e-3 for `steps` per layer (default 10,000).
  - Gather samples in windows of at most 32 samples and 512 MiB by default, refreshed every 100 steps.
  - Draw minibatches (default 8) with replacement using `seed + layer index`.
- **Parity:** each window checks Torch layer output against ORT with `atol 1e-5` and `rtol 1e-4`.
- **Checkpoints:** two types of files:
  - `completed.json`: completed layers and identity.
  - `codes_<sha>.npz`: completed-layer codes.
  - Resume after verified completed layers. Restart an interrupted layer from its beginning.
- **Activation cache** (`ActivationCacheConfig`): store each layer's sample (input, target) pairs in RAM (default 4 GiB), and place the current window on GPU (default 512 MiB with 2 GiB reserved free memory).
  - Caching affects speed, not results.

Within this quantization path, Torch is needed only for AdaRound. ORT produces the inputs and targets, and Torch computes one layer and its gradients.

## 9. Policy objects and defaults

| | `Scheme()` | `QuantizationConfig()` |
|---|---|---|
| Scope | `all` | `basic` |
| Activations | Asymmetric | Symmetric |
| Weights | Per-channel | Per-tensor |
| Range selection | Percentile 99.99 | MinMax |
| Scale policies | float / float | float / float |

Calling `quantize(model, feeds)` without a policy uses `Scheme()`. Because the objects have different defaults, specify the policy explicitly for reproducible experiments.

Experiment YAML is converted to `Scheme` by `configuration.scheme_for()`. `Scheme.config()` produces a `QuantizationConfig`, whose `scheme()` method supplies the engine dictionary. These are representations of one policy along a conversion path. Passing a `QuantizationConfig` directly skips the compact-policy conversion.

- `CalibrationConfig.percentile()` has another default: 99.999.
- Some `QuantizationConfig` fields, including certain precision, granularity and representation fields, allow only one value.

## 10. Common limitations for custom models

- Any operator missing from the role table stops quantization (section 2).
- Ambiguous gates or overlapping activation patterns are rejected (section 2).
- Non-initializer constant MatMul weights remain FP32 (section 6).
- Input-type and opset restrictions apply (section 1).
