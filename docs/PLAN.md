# FractalSync — Rhythm/Structure + Control Roadmap (Session Notes)

Date: 2026-01-31  
Source-of-truth decision: **Rust (`runtime-core`) owns audio timing / clocks and emits state; frontend consumes state.**

---

## Executive summary

We agreed to build a **two-timescale system**:

- **Slow clock (audio-hop clock, Rust):** owns rhythm + structure state (BeatNet + section/anticipation model outputs).  
  - Runs at hop cadence (~46.875 Hz for 48 kHz / 1024 hop) or a decimated cadence (e.g., 12–23 Hz).
  - Emits a **held** `SlowState` that the frontend samples at 60 FPS.

- **Fast clock (render clock, frontend):** drives high-frequency Julia micro-reactivity.  
  - Runs at ~60 FPS.
  - Reads the latest `SlowState` (and optionally advances beat phase predictively between slow updates).
  - Runs the fast controller model (or uses already-produced fast control signals) to respond to musical nuance faster than beat-level events.

Key reason for the split: we want **fast nuance reactivity** without needing beat/structure inference at 60 FPS.

---

## Big decisions (locked)

### 1) Beat tracking strategy
- **Use BeatNet** (target long-term) as the beat/downbeat/tempo tracker.
- **Run BeatNet online** for runtime behavior, and use the same BeatNet-derived signals in training so we maintain **parity** between training and real use.
- Latency is not a dealbreaker because visuals can use a **predictive oscillator** (phase + seconds-per-beat) between updates.

### 2) Section/structure detection strategy
- Goal is **real-time section detection / anticipation**.
- Use **offline structure analyzers** to annotate training songs (teacher labels) when needed.
- **SongFormer** selected as the most promising offline teacher at first glance.
- Keep **pre-chorus** as a first-class label because it enables section-aware behavior and loss terms (e.g., aesthetic constraints that depend on section).

### 3) Model architecture
We moved from a single hybrid model toward a **two-model design** because we want different polling rates:

- **Anticipator model (slow-rate):** predicts song structure state + boundary hazard.
- **Controller model (fast-rate):** uses audio micro-features + BeatNet clock + anticipator outputs as conditioning to produce control signals for Julia rendering.

Anticipator outputs flow **straight into the controller** as part of its input vector (plus other inputs).

---

## Clocks & scheduling

### Slow clock (Rust, hop cadence)
- Canonical sample rate: **48 kHz**
- Hop length: **1024 samples**
- Hop cadence: `48000 / 1024 ≈ 46.875 Hz`

Responsibilities:
- Maintain and emit `BeatClockState` (BeatNet later; placeholder oscillator immediately).
- Run the **anticipator** model at hop cadence or decimated cadence (e.g. every 2–4 hops → ~12–23 Hz).
- Emit `SlowState` frames for frontend consumption.

### Fast clock (Frontend, render cadence)
- Render cadence: ~60 FPS (`requestAnimationFrame`).
Responsibilities:
- Read/hold last `SlowState`.
- Optionally **advance beat phase predictively** between slow updates using `spb` and `dt`.
- Run fast controller at render cadence (or at hop cadence with interpolation if desired).
- Apply state smoothing/hysteresis on slow signals to prevent flicker.

---

## Runtime interface: state bus

We plan to define a stable state bus from Rust → frontend.

### Beat clock state (concept)
Fields we discussed (Rust struct):
- `t_sec`: stream time for the hop/frame
- `spb`: seconds-per-beat (tempo proxy)
- `phase`: beat phase in `[0,1)`
- `beat_count`: monotonically increasing beat index (parity anchor)
- `conf`: confidence/stability measure
- optional `downbeat_prob` if we decide to include it

### Structure/anticipation state
- `section_probs` or `section_logits` (we will standardize)
- `hazard_probs` for multiple horizons (see below)

### Combined slow state
- `SlowState = { beat: BeatClockState, structure: StructureState, features?: Vec<f32> }`

Frontend should treat `SlowState` as source-of-truth, held between updates.

---

## Anticipator model contract (v1)

### Label set
We decided to use the **same label set as SongFormer**, keeping **pre-chorus**.

SongFormer’s labels (as referenced in discussion):
- `intro, verse, chorus, bridge, inst, outro, silence, pre-chorus`

We will freeze a stable ordering in a `model_contract.json` file (see below).  
Proposed ordering for v1:

```text
labels_v1 = [
  "silence",
  "intro",
  "verse",
  "pre-chorus",
  "chorus",
  "bridge",
  "inst",
  "outro"
]
L = 8
```

### Hazard horizons
We chose beat-relative horizons:

```text
horizons_beats_v1 = [1, 2, 4, 8, 16]
K = 5
```

Semantics:
- `hazard[k]` approximates `P(section boundary occurs within next horizons_beats_v1[k] beats)`.

### Inputs (slow-rate)
- Windowed audio features aligned to slow clock: `[T_slow, F_audio]`
- BeatNet-derived beat features aligned to slow clock: `[T_slow, F_beat]`

Beat feature suggestion (time-signature agnostic):
- `sin(2π phase)`, `cos(2π phase)`, `log(spb)`, `conf`  
Optional later: `downbeat_prob`

### Outputs (slow-rate step)
- `section_logits`: shape `[L]` (or `section_probs` after softmax)
- `hazard_logits`: shape `[K]` (or `hazard_probs` after sigmoid)
- optional `macro_embedding`: shape `[E]` (if useful for controller conditioning)

Runtime conversion:
- `section_probs = softmax(section_logits)`
- `hazard_probs  = sigmoid(hazard_logits)`

---

## Controller model contract (v1)

### Inputs (fast-rate)
- Micro audio features: `[T_fast, F_fast]` (short window, tuned for nuance)
- Beat features at current frame time: `[F_beat]` (phase advanced predictively)
- Conditioning from anticipator:
  - `section_probs`: `[L]`
  - `hazard_probs`: `[K]`
  - optional `macro_embedding`: `[E]`
- Optional runtime state features (previous control outputs, previous c, etc.)

### Outputs (fast-rate)
- Your existing control vector (Δc, palette, residual gains, etc.), unchanged except for incorporating new conditioning inputs.

---

## Training pipeline plan

### Source-of-truth features
- We want runtime parity; therefore:
  - **Prefer Rust `runtime-core` features as canonical** for training and inference.
  - Training data should store feature frames on the same hop clock used at runtime (48k/1024 hop).

### Offline teacher labels
- Use SongFormer (offline) to label training tracks:
  - section boundaries `(start,end,label)`
  - section labels including **pre-chorus**
- Convert to per-hop labels `y_section[t]`.

### BeatNet parity for beat-relative targets
- Run BeatNet in a streaming-like mode (same hop settings) on training audio.
- Use BeatNet beat events to define `beat_count` over time.
- For each hop `t`, compute `Δbeats_next[t]` = beats until next SongFormer boundary.

Then define hazard targets for each horizon `h ∈ [1,2,4,8,16]`:

```text
y_hazard[t, h] = 1 if 0 < Δbeats_next[t] <= h else 0
```

### Training anticipator (supervised)
Losses:
- Section classification: cross-entropy over `y_section`.
- Hazard: BCE over `y_hazard` for each horizon.
- Optional temporal smoothing regularizer to reduce flicker.

Export anticipator to ONNX.

### Training controller (supervised / task-driven)
Controller consumes anticipator signals as conditioning.

Robustness training strategy:
- Early: teacher-forced conditioning (from ground-truth/teacher labels).
- Later: feed **anticipator predictions** (realistic noise).
- Add robustness noise:
  - random delay of anticipator state by 0–2 slow steps
  - random dropout on hazard channels
  - mild noise on probabilities

Section-aware penalties:
- Use **soft section probabilities** to weight penalties, e.g.
  - penalize connected Julia sets during choruses:
    - `L += p_chorus * penalty_connected(...)`

Export controller to ONNX.

---

## Frontend vs runtime-core feature extraction (current state & refactor plan)

### Current issue (before refactor)
Frontend feature extraction advances its internal feature buffer **each time it is polled** (currently at rAF ~60 FPS).
This makes “window size” depend on polling frequency and prevents clean slow/fast consumers.

### Refactor requirement
Decouple:
- **Producer:** appends one feature frame at a fixed cadence
- **Consumers:** read windows of frames without advancing the buffer

Frontend will increasingly become a consumer only; Rust will be source-of-truth.

---

## Known bugs/issues (to fix early)

### Feature extraction correctness bug (frontend)
We identified that spectral flux is computed in a way that can update `previousSpectrum` multiple times per frame (flux computed twice), breaking onset/flux features.

This is a **parity breaker** and should be fixed before training data generation if frontend-derived features are used anywhere.

---

## Implementation roadmap (ordered to minimize rework)

### Phase 1 — Two clocks + state bus scaffolding (foundation)
- Implement `SlowClock` in Rust with deterministic hop cadence.
- Define `BeatClockState`, `StructureState`, `SlowState` and make them serde-serializable.
- Provide a “bridge” API for the host layer to call `process_hop(hop)` and serialize `SlowState`.

Frontend:
- Stop computing model inputs in JS as the primary pipeline.
- Consume `SlowState` from Rust and hold last value for render loop.

### Phase 2 — Parity-breaking bug fixes
- Fix feature extraction correctness issues (especially any duplicated flux updates).
- Confirm stability of cadence + shapes.

### Phase 3 — Anticipator training pipeline
- Run SongFormer offline to create labels for a corpus.
- Run BeatNet offline/streaming-mode to create beat-count parity signals.
- Generate per-track artifacts:
  - `features`, `beat_state`, `y_section`, `y_hazard`
- Train anticipator; export ONNX.

### Phase 4 — BeatNet runtime integration
- Export BeatNet CRNN to ONNX.
- Implement inference + particle filter update in Rust.
- Make BeatNet the provider of `BeatClockState` in `SlowClock`.

### Phase 5 — Controller training & runtime wiring
- Train controller conditioned on anticipator outputs.
- Integrate controller model into runtime at fast cadence (60 FPS), while consuming held slow state.

---

## Contract file (recommended)
Create a versioned contract file (e.g. `runtime-core/model_contracts/v1.json`) including:
- sample rate, hop size
- label set (ordering)
- hazard horizons
- beat feature definitions
- tensor names/shapes for anticipator/controller inputs and outputs

This prevents silent breaking changes.

---

## Open items (known, but not blockers for scaffolding)
- Exact feature vector definitions and normalization scheme (we should freeze for v1).
- Whether to include `downbeat_prob` in beat features from day one.
- Exact window sizes `T_slow` and `T_fast` (can be tuned after pipeline is running).
- How/where the host layer moves PCM hops from frontend to Rust (not implemented today; we only scoped the interface).

---

## Links referenced during the session
- SongFormer repo: https://github.com/ASLP-lab/SongFormer  
- All-in-one structure analyzer (alternative teacher): https://github.com/mir-aidj/all-in-one  
- BeatNet repo: https://github.com/mjhydri/BeatNet

---

SongFormer Integration Plan (Offline Section Labeling and Boundary Annotation)

Installation: SongFormer is provided as a Python package via GitHub (ASLP-lab/SongFormer) and the Hugging Face Hub. You can install it by cloning the repository and installing requirements, including its submodules. For example:

git clone https://github.com/ASLP-lab/SongFormer.git  
cd SongFormer  
git submodule update --init --recursive  # fetch MuQ and MusicFM submodules  
conda create -n songformer python=3.10 -y && conda activate songformer  
pip install -r requirements.txt  

This ensures you have all dependencies (the authors tested on Ubuntu 22.04) ￼ ￼. Next, download the pre-trained model weights. In the repo’s src/SongFormer directory, a utility script utils/fetch_pretrained.py will pull down the SongFormer checkpoint and required SSL model weights ￼ ￼. Alternatively, you can use the Hugging Face model directly: for example, using the Transformers API with trust_remote_code=True will auto-download the model and code ￼ ￼. (No PyPI pip install is available as of now, so using the GitHub or HF Hub method is recommended.)

Running Inference on Audio: SongFormer provides a simple interface to run inference on audio files. If using the Hugging Face approach, you can load the model and call it on a file path or waveform array:

from transformers import AutoModel  
songformer = AutoModel.from_pretrained("ASLP-lab/SongFormer", trust_remote_code=True)  
songformer.eval().to("cuda")  # move to GPU if available  
result = songformer("path/to/audio/file.wav")  

This will process the audio and return the segmentation result ￼. You can also pass a NumPy array of audio samples instead of a filepath ￼. Note: SongFormer expects 24,000 Hz mono audio input ￼. If your audio is in another format (e.g. 48 kHz stereo), you should downmix to mono and resample to 24 kHz before inference to ensure the model’s internal feature extractors operate correctly. The model runs quickly (processing a full song in a few seconds on GPU) ￼, making it feasible to batch-process your training dataset offline.

SongFormer’s GitHub also includes a batch inference script (infer.sh) and Python module (infer/infer.py) for offline annotation ￼. Using infer.sh, you can point it at a directory of audio files (via an SCP list) and an output directory; it will produce an annotation file for each audio. Under the hood, this script uses multiple processes (and GPUs if available) to speed up annotation. This is convenient for labeling all training songs in one go. Ensure the --checkpoint path and --config are set to use the downloaded pre-trained model (the defaults in the script will use SongFormer’s provided checkpoint).

Input/Output Formats: The output of SongFormer inference is a list of segment boundaries with labels. In code, result will be a Python list of dictionaries like:

[ 
  {"start": 0.0, "end": 15.2, "label": "verse"},  
  {"start": 15.2, "end": 30.5, "label": "chorus"},  
  ... 
] 

Each entry marks a segment start time, end time (in seconds), and the predicted section label ￼. SongFormer’s label set includes standard sections (intro, verse, chorus, bridge, instrumental, outro, silence) and a pre-chorus category, for a total of 8 distinct labels ￼ ￼. By default, SongFormer uses these 8 classes internally (the authors note that for evaluation they sometimes merge pre-chorus into verse, but for our purposes we will keep it separate as a first-class label ￼).

For convenience, you can convert the raw output into a text file per track. The repository provides utils/convert_res2msa_txt.py to convert the result to an “MSA format” TXT file ￼. This format lists each segment’s start time and label on separate lines (with an "end" label marking the song end) – useful if you want to manually inspect or use existing evaluation code. In practice, however, you will likely take the list of segments in Python and integrate it directly into your training pipeline, as described next.

Post-processing and Alignment for Training: To use SongFormer’s output as training labels, we need to map the continuous segment timings to our model’s frame/hop indexing. Our training pipeline processes audio in fixed hops (e.g., 1024 samples @ 48 kHz, which is ~21.33 ms per hop, yielding ~46.875 frames per second). SongFormer’s segment boundaries may fall at arbitrary times, so we must convert the segment list into a frame-by-frame label sequence: essentially assign each hop frame a section label.

Steps to do this:
	•	Determine the time corresponding to each training frame index. For example, frame n (0-indexed) covers time interval [n*hop_size/sr, (n+1)*hop_size/sr). With 48 kHz and hop_size=1024, frame n starts at n*21.33 ms.
	•	Initialize an array for labels per frame (length = total number of hops for the track). Then, for each segment given by SongFormer, find the range of frame indices that fall between the segment’s start and end time, and fill those indices with the segment’s label ID. It’s useful to define a consistent index mapping for the 8 labels (e.g., “silence”=0, “intro”=1, “verse”=2, “pre-chorus”=3, “chorus”=4, “bridge”=5, “inst”=6, “outro”=7, as per your model contract) ￼. Convert the label string to the corresponding integer. Do this for all segments in the song. Any frames beyond the last segment’s end can be labeled as “end” or silence; typically SongFormer’s final segment will cover until the track end (plus an explicit end marker).
	•	Aligning to beat times (optional): In music, section boundaries usually coincide with beats (often downbeats). SongFormer’s predictions are quite precise (it achieved state-of-the-art boundary accuracy) ￼, but there may be small timing offsets (tens of milliseconds) relative to the actual beat grid. If you have beat annotations (or once you run BeatNet offline on these tracks), you can snap the boundaries to the nearest downbeat. For example, if SongFormer says a chorus starts at 30.50 s but the nearest downbeat (from BeatNet) is at 30.55 s, you might adjust the boundary to 30.55 s for consistency. This ensures that your section labels don’t switch mid-beat, which can simplify the model’s job (the anticipator can rely on beat-aligned section changes). This adjustment is not strictly necessary, but it can make the training labels more rhythmically consistent.
	•	Label smoothing at boundaries (optional): Another consideration is the abruptness of label changes. A neural model with frame-wise cross-entropy might struggle if a single frame flips from “verse” to “chorus” and the exact transition timing is uncertain. To mitigate this, you can apply a small temporal smoothing: e.g., a short window (say 0.1–0.2 s) around the boundary could be treated specially. One approach is to encode a few frames before and after the boundary with a softened label (perhaps a probabilistic mix of old and new section, instead of hard one-hot). Alternatively, you can down-weight the loss on a few transition frames. This can help the model not overly penalize slight timing shifts. If you plan to predict a “boundary hazard” or use a separate output for boundaries, this smoothing is less critical (the model can learn the uncertainty via the hazard output). In summary, label smoothing is optional, but can improve robustness at section transitions.
	•	Frame rate mapping: SongFormer’s internal feature hop is ~25 Hz (its transformer uses 0.5 s and 3 s context features ￼, and a 25 Hz intermediate feature rate in MusicFM). Our training is at 46.875 Hz. In practice, after assigning labels as above, you’ll have a label for every 46.875 Hz frame. There is no issue with the higher rate – the labels are basically piecewise constant over many frames (since sections last several seconds). Just ensure that during training, the model knows how these labels align with any input features (e.g., if you also feed beat features per hop, those align exactly in time with these labels by construction). If you ever use a different frame rate in the runtime vs. training, you would need to account for that, but here we will keep them consistent (our runtime-core will also operate its anticipator at the same hop rate as training).

Using SongFormer in the Training Pipeline: Once you have the per-hop label sequence for each training song, incorporate it into your model training. Typically, you will have your anticipator model take a sequence of features (spanning the song) and you will train it with a time-series loss (e.g., cross-entropy at each frame for section classification). The labels we derived from SongFormer serve as the “ground truth” sections (a form of teacher forcing, since SongFormer is effectively a teacher model here). It’s important to maintain parity between how these labels are produced offline and how the model will operate in real-time. That is, the runtime system will not have SongFormer; it will rely on the anticipator’s own predictions. By training on SongFormer labels, we assume SongFormer is a high-quality annotator giving us a reliable target. We must also ensure any additional signals the anticipator uses (e.g. beats, downbeats) are available and consistent in training and runtime – we address that next with BeatNet integration.

During training data preparation, you should also generate boundary targets if your model outputs something like a “boundary probability” or hazard function. Since SongFormer gives explicit segment boundaries, you can derive a boundary indicator signal from the segment list (e.g., a binary sequence with 1 at frames where a boundary occurs, 0 otherwise). In your case, you intend to predict a hazard over the next N beats ￼; you can compute this by taking the known next boundary from SongFormer and converting it to probabilities for those horizons. For example, if the next boundary is 8 beats away, then the hazard output for horizon=8 beats should be 1 at the frame corresponding to 8 beats before the boundary (and 0 before), etc. This is a more involved calculation, but it ensures the model learns not just the section labels but also how far the next transition is. All of this is based on SongFormer’s annotations, so the quality of those annotations is crucial. Fortunately, SongFormer has high accuracy in both boundary positioning and functional labeling ￼ ￼, making it a strong choice for providing ground truth.

Finally, treat the SongFormer-based labels as you would any ground truth in training: you can augment the data (if doing time stretching or other audio augmentations, you’d have to adjust the labels accordingly), and you should ensure the model doesn’t simply overfit to SongFormer’s quirks. One way to generalize is to not rely on any one teacher too heavily; but since SongFormer is currently the best available, using it as a sole teacher is reasonable. Just keep in mind any systematic biases (e.g., if SongFormer tends to mark a very short segment that a human might not label, your model will learn that convention – which is fine if it’s consistent).

In summary, SongFormer will be integrated as an offline preprocessing tool: you run each training audio through it, get a sequence of section labels (and by extension boundaries), then feed those as supervised targets into your training pipeline (with potential slight adjustments like beat alignment and smoothing). This gives your anticipator network a strong set of “ground truth” structural segments to learn from.

⸻

BeatNet Integration Plan (Real-Time Beat/Tempo Tracking in runtime-core)

Model Extraction and ONNX Export: BeatNet is an open-source Python library for joint beat, downbeat, tempo, and meter tracking ￼. We will leverage its neural network component by exporting it to ONNX for use in Rust. First, install the BeatNet package (and its deps):

pip install librosa madmom   # prerequisites  
pip install BeatNet          # installs BeatNet from PyPI  

BeatNet provides a BeatNet class (in BeatNet.BeatNet module) which wraps the entire pipeline ￼ ￼. We won’t use the pipeline as-is in production (since that’s Python), but we can use it to load the trained model. For example:

from BeatNet.BeatNet import BeatNet  
bn = BeatNet(model=1, mode='offline', inference_model='DBN', plot=[], thread=False)  

This initializes the model. Internally, this will load a pre-trained PyTorch CRNN (convolutional recurrent neural network) and either a Dynamic Bayes Net or Particle Filter for inference depending on mode. Our focus is the CRNN itself. We’ll need to extract the PyTorch nn.Module that produces beat/downbeat activations. This may involve digging into the source – likely the BeatNet class has an attribute for the neural network or you might instantiate the network from a submodule (the paper describes the network architecture clearly, so reimplementing it with the published weights is also an option if direct extraction is tricky).

Once we have the PyTorch model object (with weights loaded), we will export it to ONNX. The network expects a time series of input features per forward pass. We will not include the particle filtering in the ONNX – just the neural net that outputs activation probabilities. Assuming the network expects input of shape (batch, seq_length, feature_dim), we can create a dummy input. For example, feature_dim is 272 (explained below), and seq_length could be a few seconds worth of frames or we can use sequence length 1 and mark that dimension as dynamic. We’ll do something like:

import torch
dummy = torch.randn(1, 100, 272)  # 100 frames of 272-dim features
torch.onnx.export(bn.model, dummy, "BeatNet.onnx", 
                  input_names=["features"], output_names=["activations"], 
                  dynamic_axes={"features": {1: "seq_length"}, "activations": {1: "seq_length"}})

This will produce an ONNX graph where the sequence length is flexible. The output activations will have shape (1, seq_length, 3) – corresponding to the model’s three output classes per frame (non-beat, beat, downbeat) ￼. In the BeatNet architecture, a softmax is applied so these can be interpreted as probabilities for each frame summing to 1 (with “non-beat” meaning no beat event at that frame) ￼.

We also want the ONNX to handle the LSTM’s hidden state if possible. The BeatNet CRNN consists of convolutional layers followed by two unidirectional LSTM layers (150 units each) ￼. In a real-time context, we don’t want to re-run the LSTM from scratch for the entire song each hop – we’d prefer to carry over the LSTM state between calls. ONNX does support LSTM cells with state, but by default exporting may bake the LSTMs in a static way. To address this, we can modify the model’s forward function to accept (x, h0, c0) and output (activations, hN, cN). In PyTorch, one can do: output, (hN, cN) = self.lstm(x, (h0, c0)). We can initialize h0, c0 as zero tensors for export. By exposing these as inputs/outputs in torch.onnx.export (input_names=["features","h0","c0"], etc.), the resulting ONNX model will allow us to feed in the previous LSTM state each time and get out the new state. This way, in Rust we can maintain a persistent state vector that updates every inference call. This is a bit advanced, but it’s highly useful for true online operation. If implementing this is difficult, a simpler (but less efficient) fallback is to always run the network on a sliding window of the recent past frames at each step – however, that would re-introduce overhead and slight latency. We will aim for the stateful ONNX approach for maximum efficiency.

Input Features & Preprocessing: BeatNet’s neural net does not take raw audio; it takes precomputed spectral features. Specifically, the input is a sequence of 272-dimensional feature vectors, where each vector corresponds to one audio frame of ~46 ms ￼ ￼. The features are constructed as follows (from the paper and code):
	•	Compute a short-time Fourier transform (STFT) on the audio with a Hann window of 93 ms and a hop of 46 ms ￼. In the original implementation, this was likely done at 44.1 kHz or 22.05 kHz sample rate. For example, at 44.1 kHz, a 93 ms window is ~4096 samples and a 46 ms hop is ~2048 samples. (At 22.05 kHz, it would be 2048-sample window, 1024 hop – which is equivalent timing.) This yields a spectrogram (magnitude) for each frame.
	•	Apply a logarithmic filterbank to the magnitude spectrum. BeatNet uses a constant-Q like filterbank with 24 bands per octave from 30 Hz up to 17 kHz ￼. This results in 136 frequency bins per frame (the choice of 136 comes from covering ~6 octaves with 24 bands each, plus part of another octave – 30 Hz to 17 kHz spans about 9 octaves). The filterbank output is then converted to log-amplitude. (The code likely does np.log10(1 + spectrogram) or similar to stabilize log of zero.)
	•	Compute the first-order difference of the filterbank features over time ￼. In practice, for each frame t you take the 136-d vector and subtract the 136-d vector from frame t-1. This captures the local change in spectral energy (essentially an approximation of the percussive onset strengths). Concatenate this 136-d delta with the original 136-d log-magnitude vector to get a 272-d feature vector for frame t. For the very first frame where no previous exists, the difference can be set to zero or you can duplicate the first frame (the effect on one frame is negligible).

In our Rust runtime, we need to implement this preprocessing pipeline so that for each incoming audio frame (or small batch of frames) we produce the same kind of features BeatNet was trained on. Concretely:
	•	Resampling: Our audio is 48 kHz, but to match BeatNet’s training, it’s advisable to convert to 44.1 kHz or 22.05 kHz. This ensures the frame timing and frequency bands align. The simplest approach is to resample 48k → 44.1k for the beat analysis. This can be done with a high-quality resampler library in Rust (or even a simple linear resampler if slight quality loss in the beat tracker is acceptable). By doing so, we can use the exact STFT parameters (e.g., 4096 FFT, 2048 hop @44.1k). If resampling is not desirable, an alternative is to adjust the FFT size and hop at 48k to approximate the same time resolution (e.g., ~4464-sample window, ~2208-sample hop to get ~93/46 ms) – but non-power-of-two FFTs or unusual sizes might complicate things. We’ll assume resampling to 44.1 kHz for accuracy.
	•	STFT: Use a library like FFTW or Rust’s realfft to compute the FFT. We will have an internal buffer that accumulates audio samples. Every time we have 2048 new samples at 44.1k (which corresponds to ~46 ms), we perform an FFT on the last 4096 samples (93 ms window). This is an overlap of 50% (since 2048 hop). We apply a Hann window to the 4096-sample frame before FFT. The output is a complex spectrum; take magnitudes.
	•	Filterbank: We need to reduce the FFT magnitude (which might be 2049 bins for 4096 FFT) down to 136 bins. We can precompute a set of triangular filters or a constant-Q transform matrix. One approach: use a mel filterbank implementation but with parameters to mimic 24 bands/octave. Alternatively, since the bands are log-spaced, we could hardcode the center frequencies: for example, 30 Hz with 24 bands per octave means each band’s center frequency is f_{n+1} = f_n * 2^{1/24}. We generate frequencies up to ~17 kHz. We then sum/average FFT magnitudes in the range of each band. For simplicity, using a mel filterbank with ~136 bins from 30-17000 Hz could approximate this (mel scale is not exactly logarithmic, but close in that range). However, to be true to the paper, a constant-Q approach is implied. We might use the Madmom library’s approach if available (the BeatNet code acknowledges Madmom for “raw state space generation” ￼, which likely refers to using Madmom’s DBN or its filterbank). If implementing ourselves, we can create an array of 136 weights for each FFT bin and do a matrix multiplication. In summary, apply the filterbank to get 136 values for the frame.
	•	Log amplitude: Take \log_{10}(1 + X) or similar for each of the 136 values to compress the dynamic range. (This detail can be verified from the code; many MIR systems use \log(1 + \text{energy}).)
	•	Delta computation: Maintain the previous 136-d vector. Subtract it from the current 136-d vector to get delta. If this is the first frame, you can set delta = 0 (or simply copy the current as previous). Concatenate current + delta into a 272-d float array.

Now, feed this 272-d feature vector into the ONNX model. If you are processing multiple frames in a batch (say you accumulated 10 hops and want to process them together), you can feed a (1×10×272) tensor and get (1×10×3) output. More likely, you will process frame-by-frame for low latency: feed (1×1×272) and get (1×1×3) output each hop. If you exported the model with LSTM state inputs, you’ll also provide the last hidden state (h0, c0) and retrieve the updated state each time, storing it for the next call.

ONNX Inference I/O: In Rust, you’ll use an ONNX runtime (like Microsoft’s ONNX Runtime C API or tract or ort crate) to load BeatNet.onnx. Initialize it and prepare input tensors. The input tensor “features” will be of shape (1, T, 272). If using dynamic single-frame calls, T=1. If you included LSTM state, you’ll also feed in h0, c0 of shape (num_layers*directions, batch, hidden_size) – in our case, 2 layers * 1 direction = 2, so shape (2, 1, 150) for both h0 and c0. Initially those can be zeros. The output will be the activations (and new hN, cN if applicable). The activations for frame t might look like, for example, [0.01, 0.05, 0.94] meaning 94% confidence that this frame contains a beat (and not a downbeat, since downbeat might be indicated by a separate class). Important: We need to interpret the output correctly – based on the BeatNet paper, they output three classes: non-beat, beat, downbeat ￼. Here “beat” likely means any beat (including downbeats), and “downbeat” specifically the first beat of a bar. Some implementations use two outputs (beat vs downbeat) with beat implicitly meaning non-downbeat beat; but BeatNet’s softmax trio is [non-beat, beat, downbeat] summing to 1 ￼. We should confirm this from the code or paper: the paper implies the final layer has 3 units and softmax, so yes, that scheme. We will use these probabilities in the next stage.

Real-Time Beat Tracking Logic in Rust (Particle Filter): With per-frame probabilities coming from the neural net, we now implement BeatNet’s inference strategy in Rust. BeatNet uses a two-stage particle filter to track the beats and downbeats over time ￼ ￼. We will do the same:
	•	Stage 1: Beat Particle Filter (Tempo and Phase): This filter will track the beat interval (tempo) and the phase (position within the beat period). Each particle can be thought of as an estimate of “the next beat will occur in X frames” and “the current phase is Y”. At each new frame (hop), you advance the phase of each particle by the frame duration (for example, if a particle believes the tempo is 500 ms per beat, and our hop is ~46 ms, then phase advances by ~0.092 of a beat). The particle also might have a tempo parameter that can drift slightly (to allow tempo changes). When the phase of a particle crosses 1.0, that means the particle predicts a beat is occurring at this frame (and phase wraps around, minus 1.0).
We then weight particles by how well they explain the network’s observation. If a particle says “a beat happens now” and the neural net gave a high probability of a beat (high p_beat), that particle gets a higher weight. If a particle predicts no beat now (phase not near 1) but the network suggests a beat (p_beat high), that particle is out of sync and gets low weight. Conversely, if the network indicates no beat (high p_nonbeat) but the particle predicted one, it’s penalized. In practice: weight_update ~ particle_weight * [p_beat if particle predicts beat at this frame, else p_nonbeat]. (We may use p_nonbeat = 1 - p_beat - p_downbeat, or combine p_downbeat into the beat category for stage1 since downbeat is also a beat – an implementation detail.) We then normalize weights and resample particles (replace low-weight ones with copies of high-weight ones, adding some random jitter to tempo or phase to diversify). Over time, the particle set will converge around a certain tempo and phase that explains the input. This is exactly how BeatNet’s first stage operates and it results in robust beat tracking ￼.
From this stage, we can extract the estimated tempo and beat phase. For example, we might take the particle with the highest weight as our current best estimate: its tempo gives us seconds per beat (spb) and its phase tells us how far we are from the next beat. We can output tempo = 60/(spb) BPM as well if needed, but in the runtime state we mainly need spb. We detect an actual beat event when the best particle (or an ensemble of particles) hits phase ~1 (or 0, depending on how phase is defined). That corresponds to a beat onset in time. At that moment, we would reset phase = 0 (and all particles align, or we output a beat trigger). We then increment a beat_count. Essentially, whenever a beat is emitted by the filter, that’s the metronomic tick we’ll use for visuals or for aligning the structure model’s outputs.
	•	Stage 2: Downbeat & Meter Particle Filter: Now that we have a sense of beats, the second filter tries to group beats into measures and identify downbeats. Particles in this stage might represent hypotheses like “this is a 4/4 meter and the last beat was beat 3 of the measure” vs “it’s 3/4 and we just hit a downbeat”, etc. A simple way to implement this: each particle can carry a measure length (in beats, e.g., 4) and a position in the measure (which beat index within the bar the current beat corresponds to). When a beat event occurs (from stage 1), we advance the measure position for each particle by 1. If a particle’s position exceeds its measure length, it wraps to 1 and that means it thinks the new beat is a downbeat. We then weight the particles by the network’s downbeat probability at that moment. If the network strongly indicates a downbeat (high p_downbeat) and a particle predicts the current beat is a downbeat, that particle gets high weight; if it predicts a mid-bar beat (not a downbeat) but p_downbeat was high, it gets low weight, etc. We also weight by p_nonbeat if a particle claims no downbeat and the network agrees. (Note: p_downbeat is separate from p_beat; typically, if downbeat is high, beat is also high because a downbeat is also a beat – in the model’s output, a downbeat frame likely has most probability mass on the “downbeat” class with some on “beat”; a non-beat frame has most mass on “non-beat”. We have to handle these carefully. One approach: combine p_beat and p_downbeat for stage1 (treat any beat of any kind as “beat”), and for stage2 use the normalized ratio between downbeat vs normal beat when a beat is present.)
After weighting, we resample the downbeat particles. Over time, this filter will converge on the likely meter (e.g., it figures out if every 4th beat tends to align with high downbeat probability, etc.) ￼ ￼. The highest-weight particle in stage 2 gives us the current measure position. If, for example, it believes a downbeat happened 3 beats ago in a 4-beat bar, it means the next beat will be the downbeat of a new measure. We can output a flag or probability for downbeat: e.g., downbeat_confidence could be the sum of weights of particles that consider the current beat a downbeat. Or simply, when a beat occurs, check if the majority of particles say it’s a downbeat. If yes, mark that beat as a downbeat. This would correspond to e.g. downbeat_prob in the output state.
The meter (time signature) itself (like 3/4, 4/4, etc.) could be deduced from the best particle’s assumed measure length. BeatNet actively tracks tempo and time signature changes ￼, so our implementation could too. However, if our use-case doesn’t need to explicitly output the time signature, we might not expose it – just use it internally to know when downbeats occur.
	•	Rust Implementation: Both stages involve iterative algorithms each hop. We need to be mindful of performance, but since the hop rate is moderate (~20–50 Hz), even a few hundred particles should be fine in real-time (the original paper ran in real-time with presumably 200 particles per stage on a CPU). We will update the particle states each hop, using the probabilities from the ONNX model for that frame. We’ll likely maintain arrays of particle structs in memory, and perform operations like random sampling (for resampling) and some random perturbations (for tempo evolution or to allow occasional meter change hypotheses). The random aspects ensure the filter can recover if it loses lock or if the song’s tempo/meter changes.
	•	Output to runtime-core state: We will populate the BeatClockState struct each hop (or each output update). This includes:
	•	t_sec – the timestamp of the current audio frame (increment by hop size each time).
	•	spb – seconds per beat, which we take from the stage 1 filter’s best estimate of the period. This will update gradually as tempo changes. We might smooth it slightly (particle filters inherently smooth because they average over some uncertainty).
	•	phase – a continuous phase [0,1) of the beat. We can derive this from the time since last beat relative to spb. Essentially, phase = (current_time - last_beat_time) / spb (wrapped to [0,1)). The particle filter inherently provides a phase, but we can output a nicely smoothed phase for downstream use. This phase will reset to 0 at each beat and then increase linearly; even if we only get discrete beat events, we can interpolate phase between them assuming a stable tempo.
	•	beat_count – increment every time a beat event occurs (this count can start at 0 at the first detected beat, or 1). It provides a global beat index.
	•	conf – a confidence or stability measure. We can define this as, say, the sum of weights of particles in the dominant cluster or 1 minus the coefficient of variation of the particles’ period estimates. Essentially, if all particles agree, confidence is high; if they are scattered, confidence is low. BeatNet’s info gate mechanism looked at the entropy of the tempo distribution to decide if it can speed up processing ￼, which is analogous to measuring confidence. We will output some normalized confidence (0–1).
	•	downbeat_prob (optional) – how confident we are that the current beat is a downbeat. We can take the network’s p_downbeat at the frame of a beat, or better, the result of stage 2: for instance, the weight of particles that vote “downbeat”. This gives a value 0–1. If it’s above 0.5, we could also treat it as a binary downbeat detection. In training or usage, you might not need to output this if not required, but it was mentioned as a possible field.

These values update at the slow clock rate (the rate we run beat tracking). Initially, you might run this every hop (~46.9 Hz if using 1024@48k). However, as noted in our design, we could run it at a decimated rate if performance needs dictate. For example, running the beat tracker every 2 hops (~23 Hz) could be sufficient, since we can predict intervening beat phases. If we do decimate, the SlowState (which includes beat and structure) might update at 23 Hz while the render loop is 60 Hz – the frontend can interpolate phase between updates ￼ ￼. The BeatNet model itself was designed for ~21.7 Hz frames (46 ms), so running at ~23 Hz is very close. We should ensure that if we decimate, we adjust the internal timing accordingly (each PF update would then correspond to ~2 frames of 46ms, i.e., 92ms steps – we might accumulate the network activations over those 2 hops or take the average). For simplicity, you might choose to actually run the network every hop (46 Hz) since it’s lightweight, and only decimate the structure model (anticipator) to ~23 Hz. That way, beat timing is always as fresh as possible. In either case, it’s important that the training pipeline for the anticipator saw beat/tempo inputs at the same frequency you plan to output them at runtime. Since we plan to use BeatNet outputs in training (to maintain parity) ￼ ￼, make sure to generate them the same way (e.g., if you decide on 23 Hz updates, then downsample the training beats to 23 Hz or simulate that during training feature prep).

Real-Time Considerations: The existing BeatNet implementation does support real-time via microphone input and its internal loop, but we are essentially re-building that functionality in Rust. The official code uses PyAudio for streaming and runs the neural net + particle filter in Python in real time ￼ ￼. We can’t use Python in our production system, so we export the model to a neutral format (ONNX) and re-code the filter logic. This is an adaptation – BeatNet doesn’t natively provide an ONNX or Rust module, but it provides the science we can re-implement. One challenge is verifying that our reimplementation matches the original’s quality. We should test our ONNX and PF pipeline on some sample tracks (comparing against the Python BeatNet outputs) to ensure we haven’t introduced a regression. Pay attention to details like: the exact windowing and filterbank (to ensure the neural net sees the same scale of inputs), and the timing of how we trigger beats (the PF might output a beat a few frames after the actual event due to how particles work – the original likely handled this carefully to achieve “zero latency” beat tracking). If needed, we can introduce a very slight lookahead or smoothing to make the beats less jittery. However, the goal is an online algorithm with as low latency as possible, which BeatNet was designed for (it claims real-time, zero-lookahead performance with state-of-art accuracy) ￼.

Aligning BeatNet with Training and Runtime Audio: Finally, we clarify the hop rate and feature alignment to avoid any train/test mismatch. As discussed, BeatNet’s features are based on ~46 ms hops. Our runtime audio pipeline uses 21.33 ms hop. If we did nothing, feeding every 21 ms frame to the network would effectively double the tempo (the network would think the song is twice as fast, since it would see roughly two frames for what it expects as one frame’s worth of progression). To avoid this, we either resample or decimate. The recommended approach is to resample to 44.1 kHz and use the 2048-sample (~46ms) hop for feature extraction ￼. That means you’ll actually ignore every other 1024-sample hop in the 48k domain or combine two hops worth of audio to make one analysis frame. Concretely, you could accumulate 2048 samples of 48k audio (which is ~42.67 ms) – that’s slightly shorter than 46 ms. If you instead convert to 44.1k, 2048 samples is exactly 46.4 ms. The tiny difference (42.67 vs 46.4) over many beats would introduce drift in phase. So it’s better to truly resample. With 44.1k audio internally, you process every 2048 samples. To integrate with the 48k loop, one strategy is: as 48k samples come in, keep a separate buffer for beat analysis. For every 1024 samples at 48k, convert them to ~941 samples at 44.1k (since 44.1/48 ≈ 0.91875). You can accumulate those until you hit 2048 at 44.1k, then do an analysis step. This is a bit involved but will sync the timing. Another strategy is simpler: run the beat tracker at a slightly irregular interval such that it averages out. For example, run it every 2 hops most of the time, occasionally every 3 hops to approximate the ratio (since 46.875 Hz vs 21.7 Hz is ~2.16). This is complex to schedule and could cause jitter. Thus, resampling is cleaner.

In training, when you generated BeatNet features for input, you should mirror the same approach. Ideally, you would run the official BeatNet (or our reimplementation) on each training track to produce beat times, downbeat times, etc., and then use those as features/labels. That way, the anticipator sees realistic beat signals. We have decided to do this – using BeatNet in offline mode for training data (probably using the dynamic Bayesian network (DBN) mode for highest accuracy offline ￼, or the PF mode to mimic online). As a result, the training data has, for each frame, associated beat phase, tempo, etc., consistent with what the runtime will output. Maintaining this parity is crucial ￼: it means at runtime we can trust that the anticipator model is getting inputs in the same format and cadence as it saw during training, avoiding covariate shift.

Summary: BeatNet’s neural network will be exported to ONNX and run in the Rust audio callback, producing per-hop beat and downbeat activation probabilities. The remaining BeatNet logic – tempo tracking, beat and downbeat decision-making – will be implemented via a two-stage particle filter in Rust, which updates the beat/downbeat state on the fly. We’ll ensure the feature extraction and timing align with the training conditions (using 44.1 kHz, 46 ms frames, or an equivalent method), so that the system’s output is consistent and accurate. With this integration, the runtime-core will output a beat clock (tempo spb, beat phase, beat count) and downbeat indicator in real time, which downstream components (like the fast controller or the anticipator model for structural timing) can consume. This fulfills the requirement of a real-time beat tracker that matches the training distribution of beat features, enabling the overall system to react to music in sync and in structure-aware ways.

Sources: SongFormer usage and output format ￼ ￼; SongFormer installation guide ￼; BeatNet installation and usage ￼ ￼; BeatNet input feature extraction (93 ms window, 46 ms hop, 136-d filterbank, delta) ￼ ￼; BeatNet network outputs (beat/downbeat probabilities via softmax) ￼; Two-stage particle filtering approach for beat/downbeat inference ￼ ￼. These inform the integration steps above.
