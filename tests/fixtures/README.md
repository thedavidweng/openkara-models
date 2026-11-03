# Runtime comparison audio

`piano-cc0.wav` is a 10-second excerpt (00:40–00:50) of
“Piano test 051” by Wilfredor, published on
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Piano_test_051.wav)
under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
It is a stereo, 44.1 kHz, 16-bit PCM recording with 441,000 frames, longer
than the model's 343,980-frame inference window.

The excerpt was made from the original WAV with:

```sh
ffmpeg -nostdin -ss 40 -i Piano_test_051.wav -t 10 -map_metadata -1 \
  -c:a pcm_s16le -fflags +bitexact piano-cc0.wav
```

Its SHA-256 is
`9f5bdcaf036f6f2453048d7d809b5db39d5edf846ac5a792474049721549c2e1`.

On an Apple M3 running macOS 27.2, the stable ORT 1.27.1 CPU archive and
the official ORT 1.31.1 CPU archive produced identical raw output tensors
for all three windows of this excerpt and of a fixed-seed noise input.
The stable archive comes from catalog release `2026-08-12-001`; the model's
SHA-256 is `c3395410b1319976683bc874d97461655a9ea6089bbb0f3bd163d3829db13d02`.
The observed maximum
absolute error was `0`, below the proposed `1e-4` limit. This is one
platform measurement; it does not qualify the other targets or providers.
