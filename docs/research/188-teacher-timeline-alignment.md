# Offline alignment of tension and structure teachers

[#188](https://github.com/rhythmatician/FractalSync/issues/188) asks whether
musical tension and structural anticipation provide complementary training
targets. Keep this hypothesis in offline validation. The two labels describe
different phenomena, and the current exports alone cannot test the joint
hypothesis.

## What the teachers provide

TenseMusic combines trends in tempo, loudness, onset frequency, pitch height,
tonal tension, and roughness. Its two variants weight those trends with
attentional and memory windows. The paper reports mean held-out correlations
of 0.59 and 0.61 over 38 Western classical pieces, with shift adjustment in
the evaluation. Its trend-salience antecedent concerns perceived tension,
distinct from acoustical intensity or an estimate of felt affect.
[Barchet, Rimmele and Pelofi](https://pmc.ncbi.nlm.nih.gov/articles/PMC10798497/),
[Farbood](https://doi.org/10.1525/mp.2012.29.4.387)

At the inspected TenseMusic revision, audio loading defaults to 44.1 kHz
and trims silence. The feature pipeline fills missing features with zero,
smooths with a centered filter whose width depends on record length, and
selects a roughly 10 Hz stream. Prediction advances by an integer step set
from the inferred feature rate and the nominal 0.25 s shift. Preserve
actual timestamps, the original-to-trimmed offset, and the model variant
when comparing pieces. The normalized teacher is an offline label, since
centered smoothing and full-record transformations can see the suffix.
The source imports pitch-model, source-separation, and feature dependencies
outside the project environment. No published model rerun was performed.
[TenseMusic source at `0e704c7`](https://github.com/vivienbarchet/TenseMusic/blob/0e704c7bdb06fec2144477f7e35004931dd038ef/notebooks/tension_functions_revised.py),
[extraction notes](https://github.com/vivienbarchet/TenseMusic/blob/0e704c7bdb06fec2144477f7e35004931dd038ef/docs/index.md)

SongFormer predicts structural boundaries and section functions. Its
benchmarks report task-specific accuracy and do not label tension.
The local export probe read six files and 153 contiguous segments,
covering 313.693–833.073 seconds per file. Five files store one label per
segment and one stores top-three class probabilities per segment. These
are saved postprocessed segments. Dense soft boundary streams and
readiness targets required by #102 need the upstream time-aligned
prediction path. The existing runner can locate locally pinned source
and weights. The validator never treats an omitted top-k class as
measured zero. [SongFormer paper](https://arxiv.org/abs/2510.02797),
[local export runner](../../scripts/run_songformer_all_audio_minimal.ps1)

## Alignment probe

Use original audio sample indices as the join key, after the project
has applied its canonical audio timebase. Preserve the native teacher
timestamp, support range, preprocessing offset, end-of-window
availability, model revision, and a validity flag. A missing target
or an expired previous observation stays null. A genuine zero tension
value stays zero. The checker exercises these cases and validates both
saved SongFormer JSON forms. It accepts future-supported targets for
offline comparison. A separate timing check refuses such a record as
available at its nominal live timestamp. The checker does not test a
live Student or collect alignments for the two real teachers.
[Project timebase context](../../CONTEXT.md)

~~~powershell
.venv/Scripts/python.exe formal/checks/check_teacher_timelines.py
.venv/Scripts/python.exe formal/checks/check_teacher_timelines.py --segments-dir outputs_songformer
~~~

The second command uses local exports and prints aggregate counts. The
probe has no dependency on either upstream model. Its examples use a
synthetic tension sequence with known support and staleness.

## Use and redistribution

| Artifact | Observed permission and scope |
| --- | --- |
| TenseMusic paper | The article is open access under CC BY. Check the separate artifact terms before sharing music, code, or learned dependencies. |
| TenseMusic repository | No license file or GitHub license grant was found at the inspected revision. Keep its code, bundled music, and prediction dependencies out of the shipping project pending a documented grant. |
| SongFormer source and model card | The repository license and model README identify CC BY 4.0. Preserve attribution and verify the exact checkpoint and external SSL encoder terms. The model card metadata does not independently tag the weights with a license. |
| SongFormDB and SongFormBench | The dataset cards report CC BY 4.0. Rights to audio referenced by a corpus and to bundled third-party components need their own evidence. |

[GitHub licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository),
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/),
[SongFormer repository license at `139b2aa`](https://github.com/ASLP-lab/SongFormer/blob/139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5/LICENSE),
[model card at `a75880e`](https://huggingface.co/ASLP-lab/SongFormer/blob/a75880ed1b7375ac71860ec6c4fc9c899cf99515/README.md),
[SongFormDB](https://huggingface.co/datasets/ASLP-lab/SongFormDB),
[SongFormBench](https://huggingface.co/datasets/ASLP-lab/SongFormBench)

## Joint-supervision experiment

On authorized recordings, hold out artist, performance, and song groups
before fitting any lag or normalization. Compare the loudness baseline
with tension, structure, and their joint targets. Use unshifted scores
first and, separately, a fixed lag chosen on development data. Include
shifted and within-song shuffled nulls, quiet tension builds, loud
releases, and structural changes without tension. Report missing
coverage and disagreement rather than forcing either label to predict
the other.

The smallest Student ablation uses the same causal trunk and training
budget for baseline, structure only, tension only, and joint
supervision. Check prefix invariance of the actual live inputs and
outputs. Score teacher agreement separately from human ratings of
timing, restraint, and how well the visual trajectory explains the
phrase. For this pass, keep the two labels as independent offline
candidates and defer joint training. A rights-cleared aligned teacher
set, reproduced TenseMusic inference, and listener evidence remain
open. [Structure supervision #102](https://github.com/rhythmatician/FractalSync/issues/102),
[Student training #131](https://github.com/rhythmatician/FractalSync/issues/131)
