# Model and dataset research decision log

Research review date: 2026-10-22. Model cards and licenses can change; re-check them before adopting a dependency or deploying a commercial product.

## Candidate comparison

### audEERING Wav2Vec2 age/gender model

Source: https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender
Usage reference: https://github.com/audeering/w2v2-age-gender-how-to
Paper: https://arxiv.org/abs/2306.16962

The published model card describes an approximately 0–100 age scalar plus scores for child, female and male, using speech sampled at 16 kHz. It lists aGender, Mozilla Common Voice, TIMIT and VoxCeleb2 among its datasets. The published model license is CC BY-NC-SA 4.0, and the official usage tutorial expressly limits the public model to non-commercial use. This therefore is not approved for a general commercial product without appropriate separate rights. Its child/female/male outputs do not directly match the requested five adult/child/teen age bands and perceived vocal-presentation taxonomy. It may be an optional research baseline in a non-commercial evaluation if the data, weights and use comply with the license; it is not enabled in this build.

The model card shows a model file of approximately 363 MB. CPU inference should be benchmarked on the target computer; GPU is optional for this application's intended small-volume local use. No VGRS inference, benchmark or accuracy metric has been recorded for this candidate.

### Furence VoxProfile

Source: https://huggingface.co/furence-ai/VoxProfile

Its model card declares an MIT license and describes a TitaNet-L embedding encoder with binary male/female output and decade-age categories (10s through 60s+). It states that labeled utterances came from multiple Korean-language AIHub corpora and were split by speaker into train/validation/test. The released weights repository expects accompanying model code. Despite the more permissive listed model license, the Korean-centric training domain, binary categories and age-decade output do not validate it for English, Urdu or multilingual use, or the requested perceived-presentation interpretation. Dataset rights and model-code/data terms must also be independently checked before any broader use. It is not enabled in this build.

### Decision

Keep the default release model-free for demographic classification, and ship truthful acoustic measurements first. Do not map pitch thresholds to gender, map an unvalidated age scalar into requested bands and call it verified, or treat a raw softmax score as calibrated confidence. The chosen fallback is a transparent `model_unavailable` response until an appropriate model is acquired and speaker-independently validated.

## Dataset strategy

Before training or comparing demographic classifiers, establish for each dataset: data origin, collection consent, label definition, age measurement method, recording license and redistribution/commercial-use terms. Perceived vocal presentation labels are not equivalent to self-reported gender identity. Chronological age labels are not equivalent to how old a voice sounds. Do not redistribute dataset audio or weights into this repository without explicit rights. Record the dataset version, download source and checksum outside the source tree; keep identifiable recordings out of Git.

## Approval gates for future model integration

1. Legal/use approval for code, weights and every dataset.
2. Exact class mapping and preprocessing verified against official examples and tests.
3. Speaker-independent and duplicate-aware train/validation/test separation.
4. Macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, and calibration assessment; confidence intervals where feasible.
5. Subgroup error analysis for represented language/accent, age, recording condition and demographic groups.
6. Coverage-versus-error study for abstention; reject low-quality/out-of-distribution samples.
7. Model card, dependency pinning, CPU memory/latency benchmark, privacy review and explicit opt-in before any training-data reuse.

No research-model benchmark or VGRS accuracy is claimed by the current implementation.
