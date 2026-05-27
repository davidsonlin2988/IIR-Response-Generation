# -*- coding: utf-8 -*-
"""
Anchor-Based Response Validation
---------------------------------
Analyzes generated student responses against structured word banks to compute
the proportion of each IIR waypoint level whose responses contain anchors
associated with:
  1. Local  — concrete details explicitly stated in the text
  2. Global — world knowledge / background schema
  3. Causal — language connecting local and global reasoning
  4. Themes — multiple theme-specific banks per scenario

Improvements over v1:
  - Graduated scoring (0.0–1.0) instead of binary (0/1). Score = fraction of
    bank keywords hit, so a richer response scores higher than a thin one.
  - Negation detection. "he was not kind" no longer counts as a kindness hit.
  - Causal co-occurrence requirement. Causal language only scores if the
    response also demonstrates local or global grounding. "because he was
    tired" alone scores 0; "because Bruno was in the rain" scores normally.
  - Expanded word banks with paraphrastic expressions so responses like
    "he understood how the cat felt" are caught even without exact keywords.
  - Distinctiveness diagnostic. Checks whether anchor scores follow the
    expected ordering across waypoints and flags violations.

Matching pipeline (applied in order, stops at first hit per keyword):
  1. Exact case-insensitive substring match
  2. Lemmatized token match  (steal/stealing/stole → steal)
  3. WordNet synonym expansion  (kind → benevolent, generous, ...)
  4. Fuzzy token match via rapidfuzz  (brunno → bruno)

Required:
    pip install pandas nltk rapidfuzz

Usage:
    python anchor_analysis.py                        # uses default CSV path
    python anchor_analysis.py path/to/responses.csv  # custom path
"""

import os
import re
import sys
import pandas as pd
from datetime import datetime

# ---- Optional dependency setup -----------------------------------------------

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("WARNING: rapidfuzz not installed — fuzzy matching disabled.")
    print("         Run: pip install rapidfuzz\n")

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
    for _pkg in ["wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    LEMMATIZER = None
    print("WARNING: nltk not installed — lemmatization and synonym expansion disabled.")
    print("         Run: pip install nltk\n")

# ---- Configuration -----------------------------------------------------------

OUTPUT_DIR = "IIR_outputs"
DEFAULT_INPUT_CSV = os.path.join(OUTPUT_DIR, "IIR_scenario_script.csv")
FUZZY_THRESHOLD = 95

NEGATION_WORDS = {
    "not", "never", "no", "don't", "doesn't", "didn't", "won't",
    "can't", "couldn't", "isn't", "aren't", "wasn't", "weren't",
    "hardly", "barely", "scarcely", "without",
}

# ---- Word Banks --------------------------------------------------------------
# Expanded with paraphrastic expressions to catch meaning beyond exact keywords.
# Each bank entry can be a single word or a short phrase.

WORD_BANKS = {
    "cat": {
        "local": [
            # Names and physical details
            "bruno", "red hair", "redhead", "ginger",
            # Actions in the story
            "basketball", "bench", "meow", "purr", "lap",
            "pet", "petting", "stroked", "picked up",
            # Conditions described
            "dirty", "stinky", "yucky", "rain", "wet", "cold", "shaking",
            "street", "stray", "nobody wanted", "new home",
            # Key moments
            "put the cat down", "go away", "real friend", "take you home",
            "wash your hands", "laughed at",
        ],
        "global": [
            # Friendship and belonging
            "friend", "friendship", "true friend", "belong", "belonging",
            "accepted", "included", "excluded", "outcast",
            # Kindness and care
            "kind", "kindness", "care", "caring", "compassion", "compassionate",
            "gentle", "warm", "generous", "good person", "good heart",
            # Bullying and pressure
            "bully", "bullying", "peer pressure", "pressure", "fitting in",
            "cool", "popular", "judgment", "judged", "embarrassed",
            # Empathy expressions — paraphrastic
            "empathy", "empathetic", "sympathy", "sympathize",
            "understood how", "knew how it felt", "felt what",
            "in his shoes", "what it's like", "relate to",
            "felt for", "cared about",
            # Courage
            "courage", "brave", "stand up", "stood up",
            "did the right thing", "despite",
        ],
        "causal": [
            "because", "that's why", "which is why", "since",
            "as a result", "led to", "caused", "resulted in",
            "therefore", "thus", "due to", "in order to",
            "made him", "wanted to", "decided to", "chose to",
            "felt that", "this shows", "this means", "which means",
            "that is why", "explains why", "so he",
        ],
        "themes": {
            "loyalty": [
                "loyal", "loyalty", "faithful", "stood by", "stayed",
                "true friend", "stuck with", "committed", "devoted",
                "stood up for", "didn't leave", "had his back",
                "was there for", "didn't abandon", "despite what",
                "even though", "regardless",
            ],
            "empathy": [
                "empathy", "empathetic", "understood how", "felt for",
                "compassion", "care about", "in his shoes", "relate",
                "sympathize", "sympathy", "perspective", "feelings",
                "knew how it felt", "felt what he", "saw how sad",
                "noticed the cat", "could tell",
            ],
            "peer_pressure": [
                "pressure", "fitting in", "popular", "accepted", "rejected",
                "laughed at", "embarrassed", "embarrassment", "judgment",
                "judged", "what others think", "what they think",
                "didn't want to be", "afraid of", "scared of",
                "to fit in", "to be cool",
            ],
            "kindness": [
                "kind", "kindness", "nice", "caring", "generous",
                "warm", "gentle", "sweet", "good heart", "compassionate",
                "good person", "did something good", "helped",
                "reached out", "was there",
            ],
        },
    },

    "lying": {
        "local": [
            # Character names
            "sarah", "rachel",
            # Setting details
            "math", "test", "exam", "ms craig", "craig", "teacher",
            "40 minutes", "answer sheet", "desk",
            # Actions described
            "cheat", "cheating", "copy", "copying",
            "looked over", "looking over", "over the shoulder",
            "bubbling", "bubbled in", "whisper", "shhh",
            # Key lines from the story
            "nothing is going on", "don't tell on me",
            "best friend", "since they were toddlers",
        ],
        "global": [
            # Honesty and truth
            "honest", "honesty", "truth", "truthful", "lie", "lying",
            "integrity", "right thing", "wrong thing",
            # Friendship and loyalty
            "friend", "friendship", "best friend", "loyalty", "loyal",
            "trust", "trusted", "betray", "betrayal",
            # Moral conflict
            "right", "wrong", "moral", "ethical", "conscience",
            "dilemma", "conflict", "torn", "difficult choice",
            "hard decision", "caught between",
            # Social dynamics
            "protect", "cover", "covering for", "secret", "silence",
            "keeping quiet", "staying silent", "snitch", "tell on",
            "report", "consequence", "get in trouble",
        ],
        "causal": [
            "because", "that's why", "which is why", "since",
            "as a result", "led to", "therefore", "thus",
            "in order to", "didn't want", "to protect", "to keep",
            "to avoid", "to save", "chose not to", "decided not to",
            "afraid", "worried", "scared", "felt that",
            "this shows", "this means", "which means",
            "so she", "explains why", "that is why",
        ],
        "themes": {
            "loyalty_vs_honesty": [
                "loyal", "loyalty", "honest", "honesty", "conflict",
                "dilemma", "choice", "torn", "difficult", "hard decision",
                "caught between", "two sides", "trade-off",
                "friendship vs", "vs honesty", "wanted to be honest",
                "but also", "on one hand",
            ],
            "friendship": [
                "friend", "best friend", "relationship", "trust",
                "bond", "care", "close", "together", "long time",
                "since childhood", "known each other", "grew up",
                "didn't want to lose", "their friendship", "close friend",
            ],
            "integrity": [
                "integrity", "right thing", "wrong thing", "moral",
                "ethical", "conscience", "should have", "regret",
                "guilty", "guilt", "doing the right", "should've told",
                "knew it was wrong", "knew better",
            ],
            "peer_loyalty": [
                "loyalty", "snitch", "tell on", "rat", "protect",
                "cover", "secret", "between us", "friend first",
                "code of silence", "not say anything", "keep quiet",
                "friends don't tell", "you don't tell on",
            ],
        },
    },

    "stealing": {
        "local": [
            # Character names and objects
            "billy", "bike", "bicycle", "red bike", "beautiful bike",
            "wallet", "ice cream", "lady", "woman",
            # Numbers and specifics
            "300", "$300", "three hundred",
            # Locations and context
            "beach", "table", "outside", "ice cream shop",
            "birthday", "parents",
            # Key story moments
            "forgot her wallet", "left behind", "no i don't",
            "haven't seen", "can't find", "looking for",
            "grabbed the wallet", "opened the wallet",
        ],
        "global": [
            # Moral concepts
            "steal", "stealing", "theft", "thief", "wrong", "bad",
            "dishonest", "dishonesty", "lie", "lying",
            "moral", "conscience", "guilty", "guilt", "regret",
            # Temptation and desire
            "temptation", "tempted", "resist", "couldn't resist",
            "opportunity", "took advantage", "saw a chance",
            "gave in to", "desire", "want", "greed", "greedy",
            # Consequences
            "consequence", "result", "trouble", "punishment",
            "karma", "what goes around", "pay for it",
            "get caught", "face the consequences",
        ],
        "causal": [
            "because", "that's why", "which is why", "since",
            "as a result", "led to", "therefore", "thus",
            "in order to", "wanted to", "needed to", "to get",
            "decided to", "chose to", "couldn't resist",
            "tempted him", "made him", "led him",
            "so he", "felt that", "this shows", "this means",
            "explains why", "that is why", "saw the chance",
        ],
        "themes": {
            "temptation": [
                "tempt", "temptation", "couldn't resist", "gave in",
                "opportunity", "easy", "hard to resist", "right there",
                "just sitting there", "too good", "saw the chance",
                "took advantage", "overcome", "resisted",
            ],
            "honesty": [
                "honest", "honesty", "lie", "lying", "dishonest",
                "truth", "truthful", "admit", "confession", "confess",
                "came clean", "told the truth", "lied", "kept lying",
                "should have told", "should have admitted",
            ],
            "consequences": [
                "consequence", "result", "outcome", "regret", "guilt",
                "guilty", "punishment", "trouble", "wrong", "pay for",
                "karma", "what goes around", "face the", "get caught",
                "end up", "will regret",
            ],
            "greed": [
                "greed", "greedy", "selfish", "selfishness", "want more",
                "envy", "jealous", "desire", "not satisfied",
                "not enough", "needed more", "wanted more",
                "couldn't afford", "wanted what", "saw and wanted",
            ],
        },
    },
}

# ---- Matching utilities ------------------------------------------------------

def _lemmatize(word):
    if NLTK_AVAILABLE:
        return LEMMATIZER.lemmatize(word.lower())
    return word.lower()


def _get_synonyms(word):
    if not NLTK_AVAILABLE:
        return set()
    synonyms = set()
    for syn in wn.synsets(word.lower()):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").lower())
    return synonyms


def _tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def _is_negated(text, keyword):
    """
    Return True if a negation word appears within 4 words before the keyword.
    Prevents "not kind" from counting as a kindness hit.
    """
    neg_pattern = (
        r'\b(?:' + '|'.join(re.escape(n) for n in NEGATION_WORDS) + r')'
        r"(?:\s+\w+){0,3}\s+" + re.escape(keyword.split()[0])
    )
    return bool(re.search(neg_pattern, text, re.IGNORECASE))


def keyword_score(text, keywords, fuzzy_threshold=FUZZY_THRESHOLD):
    """
    Graduated score: returns fraction of keywords hit (0.0–1.0).
    Each keyword is checked via the 4-tier pipeline. Negated hits are excluded.
    A response mentioning 5 anchors scores higher than one mentioning 1.
    """
    if not isinstance(text, str) or not text.strip() or not keywords:
        return 0.0

    text_lower = text.lower()
    tokens = _tokenize(text_lower)
    lemmatized_tokens = set(_lemmatize(t) for t in tokens)
    hits = 0

    for kw in keywords:
        kw_lower = kw.lower()
        matched = False

        # Tier 1: exact substring
        if kw_lower in text_lower:
            matched = True

        # Tier 2: lemmatized token match
        if not matched:
            kw_lemma = _lemmatize(kw_lower.split()[0])
            if kw_lemma in lemmatized_tokens:
                matched = True

        # Tier 3: WordNet synonym expansion
        if not matched and NLTK_AVAILABLE:
            for syn in _get_synonyms(kw_lower.split()[0]):
                if syn in text_lower:
                    matched = True
                    break
                if _lemmatize(syn.split()[0]) in lemmatized_tokens:
                    matched = True
                    break

        # Tier 4: fuzzy token match (single words >= 5 chars only)
        if not matched and RAPIDFUZZ_AVAILABLE and len(kw_lower) >= 5 and " " not in kw_lower:
            for token in tokens:
                if len(token) >= 4 and fuzz.ratio(kw_lower, token) >= fuzzy_threshold:
                    matched = True
                    break

        if matched and not _is_negated(text_lower, kw_lower):
            hits += 1

    return hits / len(keywords)


def causal_reasoning_score(text, causal_keywords, local_keywords, global_keywords):
    """
    Causal score that requires grounding in local or global content.

    A response that uses causal language but shows no local or global grounding
    scores 0 — the causal connector is floating without connecting anything
    meaningful. The more grounded the response, the more the causal score counts.

    This directly addresses the concern that "because he was tired" (no story
    or world knowledge content) should not score the same as "because Bruno
    was rejected by his friends" (grounded in both).
    """
    local_s = keyword_score(text, local_keywords)
    global_s = keyword_score(text, global_keywords)
    grounding = max(local_s, global_s)

    if grounding == 0.0:
        return 0.0

    causal_raw = keyword_score(text, causal_keywords)
    return causal_raw * grounding


# ---- Per-response analysis ---------------------------------------------------

def analyze_response(text, story):
    """
    Run all word banks for a story against a single response.
    Returns {anchor_name: float} where float is a graduated 0.0–1.0 score.
    """
    banks = WORD_BANKS[story]
    results = {}

    results["local"] = keyword_score(text, banks["local"])
    results["global"] = keyword_score(text, banks["global"])
    results["causal"] = causal_reasoning_score(
        text, banks["causal"], banks["local"], banks["global"]
    )
    for theme_name, theme_words in banks["themes"].items():
        results[f"theme_{theme_name}"] = keyword_score(text, theme_words)

    return results


def build_long_df(df):
    """
    Produce one row per (respondent × story) with graduated anchor score columns.
    """
    rows = []
    for _, r in df.iterrows():
        for story in ["cat", "lying", "stealing"]:
            cols = [f"{story}_{i}" for i in range(1, 5)]
            combined = " ".join(
                str(r[c]) for c in cols if c in r.index and pd.notna(r[c])
            )
            anchors = analyze_response(combined, story)
            rows.append({
                "respondent_id": r.get("respondent_id", ""),
                "level": r.get("profile_level", ""),
                "format": r.get("format", ""),
                "experience": r.get("experience", ""),
                "story": story,
                "response_length_words": len(combined.split()),
                **{f"anchor_{k}": round(v, 4) for k, v in anchors.items()},
            })
    return pd.DataFrame(rows)


# ---- Summary & diagnostics ---------------------------------------------------

def proportion_by_level(long_df):
    anchor_cols = [c for c in long_df.columns if c.startswith("anchor_")]
    return long_df.groupby(["level", "story"])[anchor_cols].mean().round(3)


def distinctiveness_diagnostic(prop_table):
    """
    Check whether anchor scores follow the theoretically expected ordering
    across waypoints. Flags violations.

    Expected orderings (per IIR theory):
      local  : WAYPOINT1 > WAYPOINT3 > WAYPOINT2 > WAYPOINT0
      global : WAYPOINT2 > WAYPOINT3 > WAYPOINT1 > WAYPOINT0
      causal : WAYPOINT3 > WAYPOINT2 > WAYPOINT1 > WAYPOINT0

    A violation means the generated data does not match theoretical predictions
    for that anchor × story combination.
    """
    EXPECTED = {
        "anchor_local":  ["WAYPOINT1", "WAYPOINT3", "WAYPOINT2", "WAYPOINT0"],
        "anchor_global": ["WAYPOINT2", "WAYPOINT3", "WAYPOINT1", "WAYPOINT0"],
        "anchor_causal": ["WAYPOINT3", "WAYPOINT2", "WAYPOINT1", "WAYPOINT0"],
    }

    violations = []
    for story in ["cat", "lying", "stealing"]:
        story_data = prop_table.xs(story, level="story") if "story" in prop_table.index.names else prop_table

        for anchor, expected_order in EXPECTED.items():
            if anchor not in story_data.columns:
                continue
            available = [lvl for lvl in expected_order if lvl in story_data.index]
            if len(available) < 2:
                continue
            scores = story_data.loc[available, anchor]
            for i in range(len(available) - 1):
                higher = available[i]
                lower = available[i + 1]
                if scores[higher] < scores[lower]:
                    violations.append({
                        "story": story,
                        "anchor": anchor,
                        "violation": f"{higher} ({scores[higher]:.3f}) < {lower} ({scores[lower]:.3f})",
                        "expected": f"{higher} > {lower}",
                    })

    return pd.DataFrame(violations)


# ---- Main --------------------------------------------------------------------

def main(input_csv=DEFAULT_INPUT_CSV, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        print(f"       Example: python anchor_analysis.py path/to/responses.csv")
        return

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} respondents from {input_csv}")
    print(f"\nLevel breakdown:\n{df['profile_level'].value_counts().sort_index().to_string()}")
    print(f"\nFormat breakdown:\n{df['format'].value_counts().to_string()}")
    print()

    print("Running anchor analysis...")
    long_df = build_long_df(df)
    print("Done.\n")

    # ---- Main scores table ----
    prop_table = proportion_by_level(long_df)
    print("=" * 70)
    print("ANCHOR SCORES  —  by Waypoint Level × Story")
    print("=" * 70)
    print("Scores are 0.0–1.0: fraction of bank keywords hit (averaged across")
    print("responses at that level). Higher = richer expression of that anchor.\n")
    print(prop_table.to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"anchor_scores_{timestamp}.csv")
    prop_table.to_csv(output_csv)
    print(f"\nSaved to: {output_csv}")

    # ---- Distinctiveness diagnostic ----
    print("\n" + "=" * 70)
    print("DISTINCTIVENESS DIAGNOSTIC")
    print("=" * 70)
    print("Checks whether scores follow the expected ordering across waypoints.")
    print("Violations = the data does not match theoretical predictions.\n")
    violations = distinctiveness_diagnostic(prop_table)
    if violations.empty:
        print("No violations detected — scores follow expected ordering.")
    else:
        print(f"{len(violations)} violation(s) found:\n")
        print(violations.to_string(index=False))
        print("\nViolations may indicate:")
        print("  — Word banks need expansion for that anchor/story")
        print("  — The model is not distinguishing those waypoints well")
        print("  — Sample size is too small (scores are unstable)")

    # ---- Expected patterns reminder ----
    print("\n" + "=" * 70)
    print("EXPECTED PATTERNS  (based on IIR theory)")
    print("=" * 70)
    print("""
  WAYPOINT0: Low across all anchors
  WAYPOINT1: High local, low global, low causal
  WAYPOINT2: Low local, high global, low causal
  WAYPOINT3: High local, high global, high causal, high themes

  NOTE: Causal scores are weighted by local/global grounding.
  A response with no story or world knowledge content scores 0
  on causal even if it uses connecting words like "because".
    """)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV
    main(input_csv=csv_path)
