# BubbleSync-GAN + SequenceSync-GAN (combined variant)

This folder contains an experimental variant that combines **this repository's**
blob-based physical-consistency losses (Blob Count, Blob Mean Area, Blob Std
Area) with the **temporal-sequence-consistency contribution from
SequenceSync-GAN** (a separate paper/repository -- see the top-level
[README](../README.md) for a link). It is provided for anyone who wants to
reproduce the combined-loss experiments; it is not itself a separate paper.

## How this differs from the parent BubbleSync-GAN code

| | Parent folder (`../`) | This folder |
|---|---|---|
| Input | Single grayscale image | 3-frame temporal triplet, stacked as 3 channels (not RGB) |
| Discriminator | `Discriminator` (real/fake + domain class) | `Temporal_Discriminator` (real/fake + domain class + **in-order vs. shuffled** classification) |
| Data loading | One image per sample | 3 images per sample, selected in either true chronological order or randomly shuffled, with a label indicating which |
| Losses | Adversarial + reconstruction + identity + blob losses | Same, **plus** a temporal-consistency loss (`--lambda_TD`) that trains the Generator to preserve correct frame ordering through translation |
| Blob loss computation | Per single image | Per-frame (each of the 3 channels/frames scored separately, then averaged), since the translated output is a triplet, not one image |

The core blob-loss differentiability fix (see the parent README's "Fixes since
the original implementation" section) applies identically here --
`get_blobs_properties_differentiable.py` in this folder is the same fix,
adapted to be called once per frame.

## Data format

Requires an additional property beyond the parent folder's data format:
**images must contain a chronological ordering cue in their filename**
(e.g. a frame/sequence number), since the temporal loss needs to know the
true order of any 3 images it samples. The included `data_loader.py` expects
this and will raise an error for filenames it doesn't recognize, rather than
silently guessing -- see `get_temporal_sort_key()` for the exact patterns it
understands, and adjust that function to match your own dataset's naming
convention if different.

Otherwise, the same `train/domainA`, `train/domainB`, etc. folder structure
described in the parent README applies.

## Usage

Training and testing follow the same overall workflow as the parent folder
(see the top-level README's "Usage" section for CNN base classifier training,
etc.) -- only the commands below differ, since this variant has additional
required arguments (`--source_domain`, `--target_domain`, `--direction`,
`--lambda_TD`).

```bash
cd bubblesync_and_sequencesync/
bash train.sh
bash test.sh
```

`--direction` controls which domain gets translated: `B2A` translates
domain B images toward domain A's style (matching the parent folder's fixed
test-time direction); `A2B` does the reverse.

## Citation

If you use this combined variant, please cite both the BubbleSync-GAN paper
(see the top-level README) and the SequenceSync-GAN paper for the temporal-
consistency contribution used here.
