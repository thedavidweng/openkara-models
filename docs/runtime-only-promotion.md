# Runtime-only promotion: blocked design note

A proportional runtime-only promotion would let a release whose **model
weights/artifacts are byte-identical** to the authoritative stable catalog model
skip the real-stem quality suite, while still comparing the actual baseline and
candidate native archives. It does **not** assume quality is unchanged: runtime,
kernel, execution-provider and numerical changes can change outputs and
separation quality even with identical weights. Sampled equivalence bounds
behavior on the prescribed window; it is not proof of all-song quality.

**Status: not implemented and deliberately not wired to stable promotion.** No
safe end-to-end mechanism exists in this checkout without the prerequisites
below. No evidence or qualification is claimed.

## Smallest viable design

Reuse the existing candidate/stable pipeline and existing report-ref rules;
do not put runtime-only evidence in the immutable candidate manifest.

1. **Immutable candidate.** Publish the candidate release normally to the
   `candidate` channel. The manifest is immutable and contains no
   runtime-only evidence hash.
2. **Native comparison.** Reuse `scripts/compare_runtime_builds.py` (or a small
   extracted function) to compare the stable runtime archive against the
   candidate runtime archive for each supported
   `(model, target, provider)` edge. Feed exactly the same model bytes and one
   complete 343980-frame input. Record archive/model/library digests, requested
   and assigned provider, node counts, `used_fallback`, exact output shapes,
   finiteness, numerical max-abs/MSE, and both sides' load/latency/RSS/size.
   Enforce the existing release numerical/performance budgets.
3. **App integration.** OpenKara must consume the published candidate pointer
   and record, per edge, the loaded shared-library digest, provider assignment,
   node counts, finite output and exact shapes.
4. **External qualification record.** After candidate publication, commit
   `catalog/qualifications/<candidate-release-id>.json` containing the candidate
   manifest digest/generation, one immutable report ref per edge (same
   `url`/`size`/`sha256`/`report_id`/`schema_version` rules as the strict gate),
   and the immutable app-integration report ref. This record is outside the
   immutable manifest, so there is no circular hash dependency.
5. **Stable promotion.** `publish_catalog_release.py --execute` verifies the
   qualification record, every report ref/attachment and the exact candidate
   manifest/artifact bytes, then advances the stable pointer to the **same**
   candidate manifest. It never mutates the manifest.

Trust is the same as the existing strict release gate: immutable GitHub release
assets with byte length, SHA-256, report ID and schema version; published assets
are re-verified. A JSON hash alone is not asserted to prove execution; that
limitation already applies to the strict gate. Do not invent attestations or
claim more.

## Exact blockers

1. **No OpenKara integration producer.** There is no implementation of the
   per-edge app report in this checkout or the OpenKara checkout. Without it,
   actual app execution cannot be evidenced. This requires an OpenKara change
   (separate repository/session).
2. **Promotion workflow change required.** The stable publisher currently
   accepts only the strict manifest-bound report trio. Accepting an external
   qualification record is a deliberate promotion-workflow change; it must be
   reviewed and implemented, not simulated with a placeholder.
3. **Native comparison coverage.** `compare_runtime_builds.py` currently drives
   the spectral-core adapter. Covering all seven stable model artifacts (92
   edges) requires generalizing the native window measurement to the waveform
   and dual-output interfaces and to each target/provider. This is implementable
   without OpenKara changes but is not done here.
4. **Candidate packages.** Header closure is now fixed in packaging code
   (#161), but no candidate artifacts have been rebuilt. The earlier six
   archives also did not form a full+reduced counterpart set. Both must be
   rebuilt before any measurement.
5. **No real runs.** No native comparison, app integration or promotion was
   produced. Local diagnostic smoke is not qualification.

## Why the previous draft was withdrawn

The removed draft added new evidence schemas, attestation policy plumbing and a
manual assembler, yet always failed closed and still had a circular
manifest/evidence dependency. That was more machinery without a producer or a
real promotion path. This note keeps the concrete design and the blockers only.

## Concrete prerequisite list

- OpenKara: per-edge integration report producer reading the candidate pointer.
- openkara-models: generalize native old/new comparison to every model
  interface/target/provider using `compare_runtime_builds.py` primitives.
- openkara-models: commit the qualification-record reader/validator in the
  stable publisher, using existing report-ref verification.
- Build owner: rebuild all candidate archives with the complete recursive
  quoted-header closure and the correct full/reduced set.
- Then run real measurements on native runners and record actual evidence.

Until all of that exists, model changes use the strict real-corpus/frozen-baseline
gate and runtime-only promotion remains unavailable.
