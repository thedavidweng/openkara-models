
### Licensed reference bundle ingestion

`run_quality_suite.py --tier release --onnx <candidate> --runtime <archive> --corpus-dir <local-dir>`
runs **all** reviewed fixtures (PR plus release); PR mode remains the synthetic
subset. Real-audio entries require an explicit SPDX license, HTTPS provenance,
SHA256/size and `reference_stems: {"format":"npz-float32-v1", "order":["drums","bass","other","vocals"]}`.
Provision `<sha256>.npz` yourself: the suite never downloads audio. Each bundle
contains exactly `mix` float32 `[C,T]`, `stems` float32 `[4,C,T]`, scalar integer
`sample_rate`, and string array `stem_names` in that order. Mixture and stems
must be finite, sample-aligned, and sum within float32 tolerance. No resampling,
reordering or substitution is implicit.

Real fixtures must cross an inference-window boundary and contain non-silent
references with low-band energy so all seven metrics are defined. The producer
records the worst stem (and worst cross-stem leakage pair), plus chunk-boundary
MSE against ground truth at actual overlap-add boundaries. Undefined metrics
fail, rather than being replaced by finite placeholders. Separation rows carry
the exact reference bundle digest/size; stable qualification requires the same
rows as the quality producer report. Tiny generated test bundles test only this
code and are **not** licensed real-corpus or release-quality evidence.

### Exact native archive measurement

Release quality requires `--runtime <candidate archive>`, `--target` (or native
host detection), and `--provider`. Runtime quality, benchmark and full/reduced
comparison all invoke the C++ C API harness with that archive's actual library;
`ORT_RUNTIMES` is not used. Installed-wheel mode is conversion/local diagnostics
only and cannot pass the release benchmark gate. Quality/runtime producer
reports bind actual model and runtime SHA256/size as well as target/provider;
the publisher checks these against each exact compatibility edge.

New packages include the pinned C API header and its transitive quoted header
dependencies, with manifest digests. The small harness builds from these packaged headers;
`--ort-source` optionally cross-checks explicit source bytes. Historical packages
missing required headers must be rebuilt, not silently substituted. This
compiles only the harness, never a full ORT build, with at most two jobs.

`cold_load_s` sums per-window native session load; `first_window_s` times the
first native inference. `warm_median_s`/p95 and `rtf_warm` measure native Run calls
with the requested provider (sum over windows per iteration), **not** consumer
DSP or a resident-session end-to-end benchmark. `end_to_end_s`/`rtf_first`
measure the actual first full waveform pipeline, including DSP, tensor files,
process startup and per-window loads. The report records that scope explicitly;
RSS is the actual native child peak (KB), not the Python parent's peak.
Full/reduced numerical equivalence compares every raw output tensor for one
complete 343980-frame window, not just the reconstructed sum. These code tests
do not establish corpus quality, native GPU success or product performance.
