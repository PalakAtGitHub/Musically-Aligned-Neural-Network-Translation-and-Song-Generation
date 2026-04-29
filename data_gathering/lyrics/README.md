# `lyrics/` — Ground-Truth Lyric Files

This directory holds the manually-sourced Hindi/English lyric pairs that
feed `disney_lyrics_curator.py`. The whole point of this folder is that
the Hindi text in here was **professionally written by lyricists** to fit
the same melody as the English original — that's what makes this dataset
qualitatively different from FMA's IndicTrans2-generated labels.

## File format

For each song with catalog name `<song_name>` (e.g. `let_it_go_frozen`),
create two parallel plain-text files:

    <song_name>.en   ──  one English lyric line per line
    <song_name>.hi   ──  one Hindi lyric line per line

Line N in `.en` pairs with line N in `.hi`. Empty lines are ignored. The
two files **must have the same number of non-empty lines** — the curator
refuses to process a song where the line counts disagree (clear flag).

Catalog song names come from `data_gathering/disney_song_catalog.json`
(the `name` field of each entry).

## What counts as "one line"

Match the natural lyric line breaks of the song — usually one phrase or
one half-couplet per line. Roughly: wherever a singer takes a breath, or
wherever a published lyrics sheet wraps, that's a line break.

A few rules of thumb:

- Don't split a single sung phrase across two lines.
- Don't merge two distinct sung phrases into one line.
- If a chorus repeats, repeat it in both files — don't shortcut.
- Skip pure vocalisations ("ohh", "ahh", "la la la") on both sides.
- Skip spoken intro/outro narration on both sides.

The curator slices the song's melody by these line boundaries, so the
breaks matter — but they don't need to be perfect. The melody slicing is
robust to off-by-one boundary errors.

## Where to source lyrics

For Disney Hindi dubs, the realistic options are:

- **Genius** — has many Disney Hindi dub transcriptions, but coverage is
  uneven. Verify against the audio.
- **lyricstranslate.com** — strong for Bollywood-adjacent content,
  including Disney Hindi dubs.
- **Fan wikis / YouTube comments** — uneven quality; cross-check.
- **Your own transcription from the audio** — gold standard but slow.
  Recommended for songs where online sources disagree.

The English lyrics for these songs are very well documented (Disney
Wiki, Genius, official sites) — those should be near-effortless.

## Sourcing rule of thumb

If two independent sources agree on the Hindi line, use it. If they
disagree, listen to the audio and pick the version that matches what's
actually sung. If you can't tell, skip the line on both sides (just
delete that line from both files).

## Catching errors before training

Run `python -m data_gathering.disney_lyrics_curator --max 1` on a single
song first. Inspect `output/qa_report_lyrics.json` for that song. Fix
any flagged lines (syllable-ratio outliers usually indicate a
mis-paired line) before processing the rest.

## What this directory does NOT contain

- Raw audio (lives in `data_gathering/downloads/`)
- Tokenized training data (lives in `data_gathering/output/`)
- Anything that gets shipped in the released dataset

This is a working directory. Lyric files are inputs, not deliverables.
The training data shipped with the paper is tokenized integer IDs +
melody arrays only, per `DATASET_ETHICS.md`.
