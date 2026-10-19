
### Licensed reference bundle ingestion

`run_quality_suite.py --tier release --onnx <candidate> --corpus-dir <local-dir>`
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
