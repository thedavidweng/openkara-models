# Model and runtime checks

Model conversion validates the ONNX spectral-core outputs against PyTorch,
with MSE strictly below `1e-4`. The graph is checked for forbidden custom
operator domains including `com.microsoft.nchwc` and must load with official
ONNX Runtime. Native runtime checks load the model and run a complete
343980-frame window. Catalog downloads are checked against SHA-256 and size.

`scripts/compare_runtimes.py` runs the same model with stable and candidate
native runtime archives on fixed-seed noise and the CC0 recording in
`tests/fixtures/README.md`. Both inputs cross the inference window boundary.
It compares each raw output tensor in each window with maximum absolute error
at most `1e-4`. On an Apple M3, the stable 1.27.1 and official 1.31.1 CPU
runtimes produced zero error across all tested windows. Other platforms and
providers require their own measurements.

On Linux x64 the comparison also measures cold session load and warm RTF;
the candidate fails when either median exceeds 1.5 times the stable value.
The PR workflow compares the current stable runtime with official ORT 1.31.1.
The runtime publication workflow compares the actual built Linux x64
candidate with stable before publishing. Model publication runs the same
runtime comparison alongside export validation. The comparison does not
measure application end-to-end performance or SDR.

The v1 stable catalog pointer is frozen for older clients. Candidate catalog
publication verifies artifact sizes and digests but does not advance stable.
