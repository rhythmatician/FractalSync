# Offline valence/arousal teacher screen

Two official MTG affect heads ran locally on synthetic audio using the installed ONNX Runtime. This establishes a reproducible noncommercial technical baseline for [#103](https://github.com/rhythmatician/FractalSync/issues/103). The public weights for both MTG heads and for Music2Emo's MERT backbone carry noncommercial terms. No complete commercial-use stack was established among the three candidates.

## Model and license provenance

| Candidate | Exact components and native output | Licensing and local disposition |
| --- | --- | --- |
| Music2Emo v1.0 | AMAAI GitHub commit `c5f7984f78d9928b841f25bd05a47377c12c58ca` with the publisher's default `J_all.ckpt`, also in the Hugging Face snapshot `b036e59471583c3d5b30c69e63e8c7323cc36c4a`. Uses 24 kHz `m-a-p/MERT-v1-95M`, pinned here for screening at `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`, and the bundled `btc_model_large_voca.pt` chord checkpoint with key/chord features. The code pools MERT embeddings from 30 s waveform segments into a clip embedding and returns a clip-level valence/arousal pair, documented on a 1–9 scale. | GitHub code license is MIT. The publisher's Hugging Face repository containing the affect weights is marked Apache-2.0. MERT's publisher marks its separate weights CC BY-NC 4.0. The upstream BTC repository carries MIT, but this screen did not establish a separate weight grant or byte-level identity of the bundled BTC checkpoint. The existing project environment lacks Transformers and the required checkpoint stack. The publisher's MERT file alone is 377,552,987 bytes. The official Music2Emo path was left unrun. [Code license](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/LICENSE), [inference source](https://github.com/AMAAI-Lab/Music2Emotion/blob/c5f7984f78d9928b841f25bd05a47377c12c58ca/music2emo.py), [affect weights and card](https://huggingface.co/amaai-lab/music2emo/tree/b036e59471583c3d5b30c69e63e8c7323cc36c4a), [MERT card](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5), [BTC upstream](https://github.com/jayg996/BTC-ISMIR19/tree/2682317be668032e6e4b269ded36adaa2ad57df0). |
| Essentia DEAM/MusiCNN | Official `deam-msd-musicnn-2.onnx`, consuming the 200-dimensional `embeddings` output of `msd-musicnn-1.onnx`. Returns `(valence, arousal)` for each patch. MTG documents a 1–9 target scale. | MTG lists its DEAM head and MusiCNN weights under CC BY-NC-SA 4.0, with proprietary licensing available separately. Essentia library code is AGPLv3 with a separate licensing path. The Essentia TensorFlow Python package is absent locally and the publisher's PyPI release file list had no native Windows wheel. The official ONNX files loaded and ran on the installed Windows CPU provider. [DEAM metadata](https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.json), [MusiCNN metadata](https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.json), [model terms](https://essentia.upf.edu/models.html), [library licensing](https://essentia.upf.edu/licensing_information.html), [TensorFlow package](https://pypi.org/project/essentia-tensorflow/#files). |
| Essentia emoMusic/MusiCNN | Official `emomusic-msd-musicnn-2.onnx`, using the same MusiCNN embedding. MTG's model listing and the metadata `dataset` field identify emoMusic, while the metadata's `description` says DEAM. Trust the specific `dataset` field and published model listing for provisional attribution. The output is the same ordered pair and target scale as the DEAM head. | Same MTG model and backbone terms and local execution path as DEAM. Both heads share embeddings, so their results can expose disagreement without constituting two independent audio frontends. [emoMusic model listing](https://essentia.upf.edu/models.html#arousal-valence-emomusic), [emoMusic metadata](https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.json), [model terms](https://essentia.upf.edu/models.html). |

No source recordings or dataset files were needed for this screen. The DEAM and emoMusic dataset publishers describe distinct collections and temporal ratings. Their data rights, any overlap, and a licensing determination for later labels or distilled models need a separate review before using such outputs commercially. The diagnostic's use of ONNX Runtime avoids an Essentia library install. The original model terms still govern the downloaded ONNX weights. [DEAM data](https://cvml.unige.ch/databases/DEAM/), [emoMusic data](https://cvml.unige.ch/databases/emoMusic/), [MTG model license](https://essentia.upf.edu/models/LICENSE).

## Reproduce the noncommercial ONNX smoke test

Obtain these exact publisher files in an ignored local directory. The diagnostic checks SHA-256 before loading them. No weights are stored in the repo.

| Publisher export | SHA-256 |
| --- | --- |
| [msd-musicnn-1.onnx](https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.onnx) | `49668ffec47e52e94b96f45930bb46a28a1368d4bdfb5c05378fa834aca616e1` |
| [deam-msd-musicnn-2.onnx](https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.onnx) | `2cb8f33f188d73fcb651f7c065b049959166c7852f454a59c13e66fe2c092bde` |
| [emomusic-msd-musicnn-2.onnx](https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.onnx) | `f9e6ac891f07c30d8b8d53b93a760f9bd0c7c45686241b5a4386ce34cda66407` |

```powershell
rtk proxy .venv/Scripts/python.exe formal/checks/check_103_affect_models.py tmp/affect-103/models > tmp/affect-103/results.json
```

The probe uses 16 kHz synthetic mono audio, an unnormalized symmetric Hann window, 512-point power spectra at 256-sample hops, 96 Slaney-area-normalized mel bands and $\log_{10}(1+10^4x)$. These match the published feature parameters. The diagnostic keeps only complete frames starting at sample zero and complete 187-frame patches at 93-frame hops. A patch has 3.008 s of waveform support and the hop is 1.488 s. Each row records the support start and exclusive end, so any center timestamp is their midpoint. The diagnostic does not use Essentia's `FrameCutter` boundary handling or establish numerical parity with its TensorFlow pipeline. [MusiCNN feature source](https://github.com/MTG/essentia/blob/df27f949d7827a43b4afc216fd5c788454b0f8a4/src/algorithms/spectral/tensorflowinputmusicnn.cpp), [patching reference](https://essentia.upf.edu/reference/std_TensorflowPredictMusiCNN.html), [diagnostic script](../../formal/checks/check_103_affect_models.py).

The test ran 16 inputs on an Intel Core i9-14900 using Python 3.13.3, NumPy 2.3.5, librosa 0.11.0 and ONNX Runtime 1.23.2 `CPUExecutionProvider`, with one ONNX Runtime inference thread. Both heads produced a finite pair for every patch. Each of the three runs per input gave array-identical outputs in the same process. The table spans the minimum and maximum of the four control-specific median timings. It includes one backbone pass plus both heads and excludes preprocessing and session startup.

| Input length | Complete patches | Inference range |
| ---: | ---: | ---: |
| 5 s | 2 | 69–71 ms |
| 10 s | 5 | 180–251 ms |
| 20 s | 12 | 478–699 ms |
| 30 s | 19 | 654–771 ms |

On the 5 s controls, the mean native valence/arousal pairs were DEAM `(3.501, 4.379)` and emoMusic `(4.686, 4.269)` for silence, `(2.844, 2.537)` and `(4.162, 2.539)` for a 220 Hz tone, and `(2.693, 2.277)` and `(4.141, 2.338)` after tripling the tone gain. An equal-amplitude 220-to-880 Hz switch produced different means again. These are observations of model behavior on artificial inputs. Silence returning a finite pair shows why valid inference and perceptually usable affect evidence need separate handling. The gain control probes acoustic sensitivity and cannot establish whether arousal adds useful information beyond energy in music.

## Disposition

Retain both MTG heads as noncommercial research comparators with pinned assets. Keep their raw outputs separate, including disagreements. Hold Music2Emo pending a reproducible dependency/checkpoint environment and the rights decision for its full stack. Leave #103 open for local music and rehearsal-style windows, native preprocessing parity if the MTG path is retained, an energy-only comparison, cross-head calibration on song-level splits, and a versioned timeline. The present evidence does not justify a ColorIntent target or a training-only valence/arousal head. Human judgments and downstream Player effects remain separate acceptance evidence.
