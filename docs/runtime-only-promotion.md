# Runtime-only promotion

A runtime-only promotion lets a release whose **model artifacts are
byte-identical** to the current stable release promote to `stable` without
rebuilding the licensed real-stem corpus quality baseline. It does not assume
quality is unchanged: kernels, execution providers and implementation changes
can change outputs. Sampled per-edge equivalence bounds behaviour on the
prescribed window; it is not proof of all-song quality.

## Gate (implemented)

The stable publisher accepts exactly two authorisations:

1. **Strict release gate** — the existing real-corpus, frozen-baseline,
   native compatibility envelope in `scripts/validate_release_evidence.py`.
   Required for any model/weight/artifact change.
2. **Runtime-only gate** — `scripts/validate_runtime_only.py`, only when the
   strict gate fails. It requires an external qualification record at
   `catalog/qualifications/<release-id>.json` (outside the immutable
   manifest, so there is no manifest/evidence hash cycle) and verifies:

   - the record binds the exact candidate manifest bytes, the candidate
     channel pointer, the current stable pointer and the baseline manifest
     bytes; the candidate must keep the baseline stable release's trusted
     producer repository, and report assets must live in that repository;
   - every model artifact (id, archive SHA-256/size, extracted file digests
     and model semantics) is identical to the stable release;
   - runtime target/reduced families are unchanged and at least one runtime
     archive differs, and every stable supported edge (mapped to the candidate
     runtime family) is still supported — dropping or reclassifying support
     fails closed;
   - immutable native-comparison reports together cover every supported
     compatibility edge exactly once (one report per target/provider group). Each edge binds the candidate and
     stable runtime digests and the model digest, and records measurements
     from the actual archives on that native host: requested and assigned
     provider, no whole-session fallback, zero CPU fallback nodes on
     accelerated providers, provider node count, exact output shapes,
     finiteness, cold load, first-window and warm latency, peak RSS,
     archive/installed size, numeric old-vs-new max-abs and MSE;
   - immutable OpenKara app-integration reports cover the same edges with the
     loaded shared-library digest, provider assignment, node
     counts, output shapes, finite output and a finite four-stem audio
     pipeline run;
   - the existing `quality/budgets.json` non-regression budgets apply: numeric
     equivalence (`mse`, `max_abs_error`), `rtf_warm`, `cold_load_s`,
     `warm_median_s`, `peak_rss_kb`, `installed_size`, plus the equality
     checks (`shape_match`, `onnx_has_nan`, `onnx_has_inf`). The validator
     recomputes them from the report values; it does not trust a `passed` flag.

Missing, duplicate, stale, skipped, fallback-substituted, non-finite or
partially covered evidence fails closed. Report bytes are downloaded and
verified by byte length, SHA-256, report ID and schema version, and report
assets must live in the catalog repository's own immutable releases.

### What this establishes (and does not)

Hashes, refs and schemas establish that the record and report bytes are
immutable, complete, internally bound and reproducible for review. They do
**not** cryptographically prove that a producer executed inference or the app:
both native and app evidence remain self-reported JSON until a trusted
attestation exists. Qualification therefore also requires repository review
and the CI pointer-transition guard; do not describe a passing hash check as
proof that measurement or integration occurred. The native harness result is
produced by the runtime archive's C API (its packaged header closure, or the
committed pin for a historical archive), never by an installed Python wheel.

The producer verifies the actual host before measuring: the OS and normalized
architecture must match the target, and a Rosetta-translated macOS process is
rejected (so an Apple Silicon host cannot masquerade as native x86_64). Host
facts are encoded in the report and checked for internal consistency by the
validator. They remain self-declared data, not hardware attestation, so they
reduce accidental mislabelling rather than spoofing.

The committed qualification record remains valid after promotion: CI
revalidates it in historical mode. Promotion itself requires the
qualification-time stable/candidate pointers, and a pull request that edits
`catalog/channels/stable.json` directly must pass
`scripts/verify_stable_pointer_change.py`, which requires a committed
qualification record (or strict evidence) bound to the promoted-from stable
release and verifies the release is published and the remote pointer manifest asset's
byte size and SHA-256 against the committed manifest.

## Producing evidence

Native comparison for one target on its native host (repeat per target and
provider set):

```sh
python scripts/run_runtime_only_comparison.py \
  --candidate-release catalog/releases/<candidate>.json \
  --baseline-release catalog/releases/<stable>.json \
  --artifacts-dir /path/to/model-and-runtime-archives \
  --target aarch64-apple-darwin \
  --report runtime-only-native-aarch64-apple-darwin.json
```

The artifacts directory must contain each referenced archive under its
manifest filename; the producer verifies archive digests before measuring,
extracts the single ONNX graph per model, verifies every declared extracted
file digest, and feeds the same bytes to both runtimes. Every supported edge
for the target must be measured; providers cannot be filtered to avoid a
failure.

Published 1.27.1 runtime archives predate C API header packaging, so the
producer compiles the measurement harness from the committed
`ort/pinned-headers.json` closure for the archive's exact upstream commit and
API level. Those bytes are fetched from that immutable commit, verified against
the pinned size/SHA-256, and checked against the runtime's build manifest; the
runtime archive itself is never rewritten. Rebuilt candidate archives package
their own header closure (#161) and are measured from the archive.

OpenKara must produce the per-edge app-integration report by consuming the
published candidate pointer (schema:
`catalog/schema-runtime-only-v1.json#/$defs/app_report`). It must be implemented
in the OpenKara repository; it is currently not verified or available in this
checkout, so this gate cannot pass until that producer exists. The producer
must, for every supported
`(model, runtime, target, provider)` edge of the candidate manifest:

- record the candidate `release_id`, generation and manifest SHA-256 it
  actually loaded;
- record the OpenKara version and full commit;
- load the candidate runtime's exact shared library and record its filename
  and SHA-256;
- run full inference through the app's own audio path with the requested
  provider, recording requested/assigned provider, no whole-session fallback,
  provider and total node counts, exact output shapes and finiteness;
- process at least one full 343980-frame window at 44100 Hz and record four
  finite stems in `audio_pipeline`;
- leave `checks` empty only on complete success.

Each report is one immutable JSON asset for a target/provider group; all
reports must together cover every supported edge exactly once.

After publishing the candidate to the `candidate` channel and uploading the
reports as immutable assets:

```sh
python scripts/assemble_runtime_only_qualification.py \
  --release <candidate> --baseline-release <stable> \
  --native-report <immutable-url>=<local-native-report> ... \
  --app-report <immutable-url>=<local-app-report> ...
```

The assembler runs the full gate against the exact bytes and writes nothing on
failure. Promotion then uses the existing publisher:

```sh
python scripts/publish_catalog_release.py --release <candidate> --channel stable --execute
```

The publisher verifies the record and reports, then advances the stable pointer
to the **same** candidate manifest. It never mutates the candidate manifest.
After promotion, use `scripts/validate_runtime_only.py --historical` (also run
by catalog CI) to revalidate the committed record; this does not require the
current pointers to still be the qualification-time ones.

## Current status

The gate and producer are implemented and fail closed. **No runtime-only
promotion evidence exists yet**, so runtime-only promotion cannot currently
authorise a release. Exact prerequisites:

1. **Rebuild the candidate runtime set.** The ORT 1.31.0 candidate artifacts
   from run 40999157757 are six reduced-only archives and predate the complete
   recursive header closure (#161); for example the Windows header includes
   `onnxruntime_error_code.h`, which is not packaged. The full/reduced
   counterpart set for the 11 shipped runtime families has not been rebuilt.
2. **Run the native comparison on every target/provider** (macOS CoreML+CPU,
   Linux XNNPACK+CPU, Windows DirectML+CPU) on hardware where the requested
   provider actually initializes and executes, and attach the reports. A
   skipped or failed provider (for example the observed DirectML C0262002
   display-adapter error) is a failure, not a pass.
3. **OpenKara app-integration producer.** No implementation exists in the
   OpenKara checkout; without it the app-report coverage check fails.
4. **Publish the candidate and commit the record.**

Until 1–4 exist, model changes use the strict gate and runtime-only promotion
remains unavailable. Do not create a placeholder record.

## Ensemble note

The published `htdemucs_ft` spectral artifacts were merged by averaging all
four sub-models. The pinned Demucs configuration is the 4×4 identity weight
matrix: each source comes from its own sub-model. The exporter and both
PyTorch references now apply the per-source weights. Existing published
artifacts predate that correction and would need a strict-gated rebuild; it
does not affect a runtime-only promotion of those same published bytes.
