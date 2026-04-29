"""
Dataset Search — Find existing multilingual song translation datasets
======================================================================
Curated list of known academic datasets and resources relevant to
musically-aligned translation research. Also provides search utilities
for discovering new datasets on HuggingFace, Zenodo, and Papers With Code.

Usage:
  python -m data_gathering.dataset_search
  python -m data_gathering.dataset_search --search "hindi song lyrics"
"""

import json
from pathlib import Path
from typing import List, Dict

# ============================================================================
# Known datasets relevant to musically-aligned song translation
# ============================================================================

KNOWN_DATASETS = [
    {
        "name": "DALI — Dataset of Automatic Lyrics and audio Integration",
        "url": "https://github.com/gMusic/DALI",
        "description": (
            "5,358 songs with time-aligned lyrics at word, line, and paragraph "
            "level. Audio from YouTube, lyrics from multiple sources. "
            "English-dominant but multi-language. Great for lyrics-melody alignment."
        ),
        "languages": ["en", "es", "fr", "de", "it", "pt"],
        "size": "~5,358 songs",
        "relevance": "HIGH — time-aligned lyrics provide alignment ground truth",
        "license": "Research-only (CC BY-NC-SA 4.0)",
        "citation": "Meseguer-Brocal et al., ISMIR 2018",
    },
    {
        "name": "NUS-48E Sung and Spoken Lyrics Corpus",
        "url": "https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/",
        "description": (
            "48 English songs sung by 12 subjects, with phoneme-level alignment. "
            "Includes both sung and spoken versions of same lyrics."
        ),
        "languages": ["en"],
        "size": "48 songs × 12 singers",
        "relevance": "MEDIUM — phoneme alignment but English-only",
        "license": "Research use",
        "citation": "Duan et al., 2013",
    },
    {
        "name": "Jamendo Lyrics — Multilingual Sung Lyrics",
        "url": "https://github.com/f90/jamendolyrics",
        "description": (
            "20 songs in English, French, German, Spanish with word-level "
            "time-aligned lyrics. Useful for training lyrics transcription systems."
        ),
        "languages": ["en", "fr", "de", "es"],
        "size": "20 songs (80 annotations)",
        "relevance": "MEDIUM — multilingual but no Hindi, small size",
        "license": "CC BY-NC-SA",
        "citation": "Stoller et al., ISMIR 2019",
    },
    {
        "name": "MusicNet — Annotated Music Dataset",
        "url": "https://zenodo.org/record/5120004",
        "description": (
            "330 classical recordings with note-level annotations "
            "(pitch, instrument, onset/offset). No lyrics but rich "
            "melody annotation that could complement vocal melody extraction."
        ),
        "languages": [],
        "size": "330 recordings",
        "relevance": "LOW — instrumental only, but useful for melody features",
        "license": "CC BY 4.0",
        "citation": "Thickstun et al., ICLR 2017",
    },
    {
        "name": "Hindi Song Lyrics Dataset (Kaggle)",
        "url": "https://www.kaggle.com/datasets/saurabhshahane/hindi-song-lyrics",
        "description": (
            "~10,000 Hindi Bollywood song lyrics scraped from LyricsMint. "
            "Text only (no audio/melody), but useful for Hindi lyric "
            "language modeling and syllable pattern analysis."
        ),
        "languages": ["hi"],
        "size": "~10,000 songs",
        "relevance": "MEDIUM — Hindi text patterns, no audio alignment",
        "license": "CC0 / Public Domain",
        "citation": "Community dataset",
    },
    {
        "name": "MUSDB18 — Music Source Separation",
        "url": "https://sigsep.github.io/datasets/musdb.html",
        "description": (
            "150 songs with isolated stems (vocals, drums, bass, other). "
            "Gold standard for source separation evaluation. Useful for "
            "validating Demucs output quality."
        ),
        "languages": ["en"],
        "size": "150 songs (~10 hours)",
        "relevance": "MEDIUM — separation ground truth, English only",
        "license": "Research use",
        "citation": "Rafii et al., 2017",
    },
    {
        "name": "Lyrics Translation Dataset (Ou et al.)",
        "url": "https://aclanthology.org/2024.lrec-main.1/",
        "description": (
            "Parallel song lyrics corpus for translation research. "
            "English-Chinese pairs with melody alignment annotations. "
            "Closest existing work to musically-aligned translation."
        ),
        "languages": ["en", "zh"],
        "size": "~200 song pairs",
        "relevance": "VERY HIGH — directly related approach, different language pair",
        "license": "Research use",
        "citation": "Ou et al., LREC-COLING 2024",
    },
    {
        "name": "Wasabi Song Corpus",
        "url": "https://github.com/micbuffa/WasabiDataset",
        "description": (
            "2.1M songs with metadata, lyrics, and NLP annotations. "
            "Includes emotion, topic, and structure tags. "
            "Useful for lyric-level features at scale."
        ),
        "languages": ["en", "multi"],
        "size": "2.1M songs",
        "relevance": "MEDIUM — large scale lyrics but no audio alignment",
        "license": "Research use",
        "citation": "Meseguer-Brocal et al., 2019",
    },
    {
        "name": "FMA — Free Music Archive",
        "url": "https://github.com/mdeff/fma",
        "description": (
            "106,574 tracks with metadata. You already use FMA-medium "
            "in your pipeline. Listed here for completeness."
        ),
        "languages": ["en", "multi"],
        "size": "106K tracks (medium: 25K)",
        "relevance": "ALREADY IN USE — your fma_data_builder.py uses this",
        "license": "CC licenses (varies by track)",
        "citation": "Defferrard et al., ISMIR 2017",
    },
]


# ============================================================================
# Search utilities
# ============================================================================

def print_known_datasets(filter_relevance: str = None):
    """Print the curated list of known datasets."""
    print(f"\n{'='*70}")
    print("  Known Datasets for Musically-Aligned Translation Research")
    print(f"{'='*70}\n")

    for ds in KNOWN_DATASETS:
        if filter_relevance and filter_relevance.upper() not in ds["relevance"].upper():
            continue
        print(f"  {ds['name']}")
        print(f"  URL: {ds['url']}")
        print(f"  {ds['description']}")
        print(f"  Languages: {', '.join(ds['languages']) if ds['languages'] else 'N/A'}")
        print(f"  Size: {ds['size']}")
        print(f"  Relevance: {ds['relevance']}")
        print(f"  License: {ds['license']}")
        print(f"  Citation: {ds['citation']}")
        print()


def search_huggingface(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search HuggingFace Hub for datasets matching a query.

    Requires: pip install huggingface-hub
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = api.list_datasets(
            search=query,
            sort="downloads",
            direction=-1,
            limit=max_results,
        )
        datasets = []
        for ds in results:
            datasets.append({
                "id": ds.id,
                "downloads": ds.downloads,
                "tags": ds.tags[:5] if ds.tags else [],
                "url": f"https://huggingface.co/datasets/{ds.id}",
            })
        return datasets
    except ImportError:
        print("  huggingface-hub not installed. pip install huggingface-hub")
        return []
    except Exception as e:
        print(f"  HuggingFace search error: {e}")
        return []


def search_all(query: str = "multilingual song lyrics translation"):
    """Run searches across all sources and print consolidated results."""
    print(f"\nSearching for: '{query}'\n")

    # Known datasets
    print_known_datasets()

    # HuggingFace
    print(f"\n{'='*70}")
    print(f"  HuggingFace Hub results for '{query}'")
    print(f"{'='*70}\n")
    hf_results = search_huggingface(query)
    if hf_results:
        for ds in hf_results:
            print(f"  {ds['id']}")
            print(f"    Downloads: {ds['downloads']:,}")
            print(f"    Tags: {', '.join(ds['tags'])}")
            print(f"    URL: {ds['url']}")
            print()
    else:
        print("  No results (or huggingface-hub not installed)\n")

    # Suggestions for manual search
    print(f"\n{'='*70}")
    print("  Additional manual search suggestions")
    print(f"{'='*70}\n")
    print("  1. Papers With Code:")
    print(f"     https://paperswithcode.com/search?q={query.replace(' ', '+')}\n")
    print("  2. Zenodo:")
    print(f"     https://zenodo.org/search?q={query.replace(' ', '+')}\n")
    print("  3. Google Dataset Search:")
    print(f"     https://datasetsearch.research.google.com/search?query={query.replace(' ', '+')}\n")
    print("  4. ISMIR proceedings (music information retrieval):")
    print("     https://ismir.net/resources/\n")
    print("  5. ACL Anthology (NLP/translation):")
    print(f"     https://aclanthology.org/search/?q={query.replace(' ', '+')}\n")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for existing multilingual song translation datasets"
    )
    parser.add_argument(
        "--search", type=str,
        default="multilingual song lyrics translation",
        help="Search query"
    )
    parser.add_argument(
        "--known-only", action="store_true",
        help="Only show curated known datasets"
    )
    parser.add_argument(
        "--high-relevance", action="store_true",
        help="Filter to HIGH relevance datasets only"
    )
    args = parser.parse_args()

    if args.known_only:
        relevance = "HIGH" if args.high_relevance else None
        print_known_datasets(filter_relevance=relevance)
    else:
        search_all(args.search)
