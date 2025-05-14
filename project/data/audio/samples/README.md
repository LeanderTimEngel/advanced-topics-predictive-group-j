# Audio Samples

This directory contains audio samples from the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song).

## File Naming Convention

RAVDESS files follow this naming convention:
```
03-01-[emotion]-01-01-01-[actor].wav
```

Where:
- `03` indicates speech (not song)
- `01` indicates full-sentence statement (not question)
- `[emotion]` is the emotion identifier:
  - `01` = neutral
  - `03` = happy
  - `04` = sad
  - `05` = angry
  - (and others not included in our sample set)
- `01-01-01` represent standard intensity, normal statement
- `[actor]` is the actor identifier:
  - `01` = female actor
  - `11` = male actor

## Current Samples

This directory contains 8 audio files representing 4 emotions from 2 actors:

| Filename | Emotion | Actor Gender |
|----------|---------|--------------|
| 03-01-01-01-01-01-01.wav | Neutral | Female |
| 03-01-01-01-01-01-11.wav | Neutral | Male |
| 03-01-03-01-01-01-01.wav | Happy | Female |
| 03-01-03-01-01-01-11.wav | Happy | Male |
| 03-01-04-01-01-01-01.wav | Sad | Female |
| 03-01-04-01-01-01-11.wav | Sad | Male |
| 03-01-05-01-01-01-01.wav | Angry | Female |
| 03-01-05-01-01-01-11.wav | Angry | Male |

## Usage

To download these samples (if they're not already present), run:

```bash
python project/data/audio/download_samples.py
```

## Reference

The RAVDESS dataset is described in:

Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391 