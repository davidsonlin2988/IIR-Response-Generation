# -*- coding: utf-8 -*-
"""
Hybrid Anchor Analysis  —  Cluster Keyword + Semantic Exemplar Scoring
-----------------------------------------------------------------------
Three scoring layers, combined into a composite:

  1. Cluster keyword scoring
     Word banks are structured as {cluster: [keywords]}.
     Score = clusters_hit / total_clusters.
     Immune to bank-size bias — adding synonyms to a cluster does NOT
     inflate the denominator. 4-tier matching pipeline per keyword:
     exact substring → lemmatized token → WordNet synonyms → fuzzy.
     Negation detection; causal score requires local/global grounding.

  2. Semantic exemplar scoring
     Two hand-written exemplar responses per (story × waypoint × question).
     Each response is embedded; cosine similarity to exemplars gives:
       - sim_WAYPOINTN   : similarity to each waypoint's exemplars
       - predicted_waypoint : argmax (which level does this look most like?)
       - semantic_quality   : 0–1 weighted score (W3=1.0, W0=0.0)

  3. Composite scoring
     keyword_composite  = weighted(local, global, causal, themes)
     overall_composite  = 0.4 * keyword + 0.6 * semantic

Granularity:
  - Per question (q1–q4 individually)
  - By question type (inference: q1+q3, metacognitive: q2+q4)
  - Per story aggregate

Embedding backends (tried in order):
  1. Vertex AI  text-embedding-004  (requires gcloud auth + PROJECT_CODE env var)
  2. sentence-transformers  all-MiniLM-L6-v2  (local, no auth — pip install sentence-transformers)

Output CSVs (all timestamped):
  hybrid_detail_*.csv            per-question rows with all scores
  hybrid_story_summary_*.csv     mean by level × story
  hybrid_qtype_summary_*.csv     mean by level × story × question_type
  hybrid_perquestion_summary_*.csv  mean by level × story × question
  hybrid_waypoint_accuracy_*.csv    semantic prediction accuracy per level

Usage:
    python hybrid_anchor_analysis.py
    python hybrid_anchor_analysis.py path/to/responses.csv
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("WARNING: rapidfuzz not installed — fuzzy matching disabled.  pip install rapidfuzz")

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
    print("WARNING: nltk not installed — lemmatization and synonym expansion disabled.  pip install nltk")

# ── Embedding backend ──────────────────────────────────────────────────────────

EMBEDDING_BACKEND = None
_embed_model = None


def _init_embedding_backend():
    global EMBEDDING_BACKEND, _embed_model

    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        project = os.environ.get("PROJECT_CODE")
        if project:
            vertexai.init(project=project, location="us-central1")
        _embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        _test = _embed_model.get_embeddings(["warmup"])
        _ = _test[0].values
        EMBEDDING_BACKEND = "vertexai"
        print("Embedding backend: Vertex AI text-embedding-004")
        return
    except Exception as e:
        print(f"Vertex AI embedding unavailable ({type(e).__name__}: {e})")
        print("Trying sentence-transformers fallback...")

    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        EMBEDDING_BACKEND = "sentence_transformers"
        print("Embedding backend: sentence-transformers all-MiniLM-L6-v2")
        return
    except ImportError:
        pass

    print("WARNING: No embedding backend found. Semantic scoring disabled.")
    print("  Install one:  pip install google-cloud-aiplatform   (Vertex AI)")
    print("                pip install sentence-transformers      (local)")
    EMBEDDING_BACKEND = None


def embed_texts(texts):
    """
    Embed a list of strings. Returns list of np.ndarray (or list of None on failure).
    Batches Vertex AI calls in groups of 5 to stay within API limits.
    """
    if EMBEDDING_BACKEND is None or not texts:
        return [None] * len(texts)

    if EMBEDDING_BACKEND == "vertexai":
        results = []
        for i in range(0, len(texts), 5):
            batch = texts[i : i + 5]
            try:
                embs = _embed_model.get_embeddings(batch)
                results.extend([np.array(e.values, dtype=float) for e in embs])
            except Exception as exc:
                print(f"\n  Embedding batch {i//5 + 1} failed: {exc}")
                results.extend([None] * len(batch))
        return results

    if EMBEDDING_BACKEND == "sentence_transformers":
        vecs = _embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        return [vecs[i] for i in range(len(vecs))]

    return [None] * len(texts)


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR    = "IIR_outputs"
DEFAULT_INPUT = os.path.join(OUTPUT_DIR, "IIR_scenario_script.csv")
FUZZY_THRESHOLD = 75
WAYPOINTS = ["WAYPOINT0", "WAYPOINT1", "WAYPOINT2", "WAYPOINT3"]

NEGATION_WORDS = {
    "not", "never", "no", "don't", "doesn't", "didn't", "won't",
    "can't", "couldn't", "isn't", "aren't", "wasn't", "weren't",
    "hardly", "barely", "scarcely", "without",
}

# Composite weights
W_KEYWORD  = 0.40
W_SEMANTIC = 0.60
# Keyword sub-weights (must sum to 1.0)
W_LOCAL  = 0.20
W_GLOBAL = 0.25
W_CAUSAL = 0.30   # higher because causal already encodes local/global grounding
W_THEMES = 0.25
# Semantic quality weights per waypoint (W0=0 → W3=1)
SEM_WEIGHTS = {"WAYPOINT0": 0.0, "WAYPOINT1": 1/3, "WAYPOINT2": 2/3, "WAYPOINT3": 1.0}

# Waypoint prediction confidence thresholds.
# A prediction is only made when BOTH conditions are met:
#   PRED_MIN_SIM    — the top similarity score must clear this floor.
#                     Filters out responses too dissimilar from all exemplars.
#   PRED_MIN_MARGIN — the top score must beat the second-best by at least this.
#                     Filters out near-ties where no level is a clear winner.
# Responses that fail either check get predicted_waypoint = None ("uncertain").
# These are excluded from the accuracy table so they don't drag down the numbers.
PRED_MIN_SIM    = 0.65   # adjust upward to require stronger similarity
PRED_MIN_MARGIN = 0.03   # adjust upward to require clearer separation between levels

# ── Cluster Word Banks ─────────────────────────────────────────────────────────
# Structure: {cluster_name: [keywords]}
# Score = clusters_hit / total_clusters
# Adding more synonyms to a cluster does NOT change its weight in the score.

WORD_BANKS = {
    "cat": {
        "local": {
            "identity":       ["bruno", "red hair", "redhead", "ginger", "red headed"],
            "cat_state":      ["dirty", "stinky", "yucky", "rain", "wet", "cold", "shaking",
                               "stray", "street", "nobody wanted", "nobody wanted him",
                               "no one wanted him", "alone", "by himself", "kitty", "kitten"],
            "interactions":   ["basketball", "bench", "meow", "purr", "lap",
                               "pet", "petting", "stroke", "picked up",
                               "bent down", "petted the cat", "petted it", "pet the cat",
                               "stopped petting", "play with"],
            "key_moments":    ["put the cat down", "go away", "real friend",
                               "take you home", "new home", "take him home",
                               "took the cat home", "took him home", "walking away"],
            "social_context": ["laughed at", "wash your hands", "hahaha", "being mean",
                               "boys laughing", "shouted", "laughing at him",
                               "laughed at the boy", "crying", "cry", "being called", "called"],
        },
        "global": {
            "friendship":    ["friend", "friendship", "true friend", "belong",
                              "accepted", "included", "excluded", "outcast",
                              "lonely", "loneliness", "left out", "fit in", "fitting in",
                              "inner beauty", "left alone"],
            "kindness":      ["kind", "kindness", "care", "caring", "compassion",
                              "compassionate", "generous", "good heart", "good person",
                              "welcomed", "take care"],
            "peer_pressure": ["bully", "bullying", "peer pressure", "fitting in",
                              "cool", "popular", "judgment", "embarrassed", "embarrassment",
                              "mean", "made fun of", "appearances", "authenticity"],
            "empathy":       ["empathy", "empathetic", "sympathy", "sympathize",
                              "understood how", "felt for", "relate", "in his shoes",
                              "what it feels like", "knew how it felt"],
            "courage":       ["courage", "brave", "stand up", "stood up",
                              "did the right thing", "despite", "regardless"],
        },
        "causal": {
            "connectors":  ["because", "that's why", "which is why", "since",
                            "as a result", "therefore", "thus", "due to",
                            "even though", "I know that"],
            "motivation":  ["made him", "wanted to", "decided to", "chose to",
                            "felt that", "in order to"],
            "explanation": ["this shows", "this means", "which means",
                            "that is why", "explains why", "so he",
                            "in the comic", "in the text", "in the story"],
        },
        "themes": {
            "loyalty":       ["loyal", "loyalty", "faithful", "stood by", "stayed",
                              "true friend", "stuck with", "didn't abandon",
                              "despite what", "even though", "regardless"],
            "empathy":       ["empathy", "empathetic", "understood how", "felt for",
                              "compassion", "saw how sad", "noticed the cat",
                              "could tell", "knew how"],
            "peer_pressure": ["pressure", "fitting in", "popular", "laughed at",
                              "embarrassed", "what others think", "to fit in",
                              "to be cool", "scared of being", "made fun of", "appearances"],
            "kindness":      ["kind", "kindness", "nice", "caring", "generous",
                              "warm", "gentle", "helped", "reached out", "was there"],
            "acceptance":    ["accepted", "acceptance", "welcomed", "belong", "included",
                              "felt included", "fit in", "fitting in"],
            "loneliness":    ["lonely", "loneliness", "felt lonely", "alone", "by himself",
                              "no one wanted", "nobody wanted", "left out", "left alone"],
        },
    },

    "lying": {
        "local": {
            "characters":    ["sarah", "rachel", "ms craig", "craig"],
            "setting":       ["math", "test", "exam", "40 minutes", "answer sheet", "desk"],
            "cheating_act":  ["cheat", "cheating", "copy", "copying",
                              "over the shoulder", "bubbling", "bubbled in",
                              "copied his answer sheet", "copied his paper",
                              "looking over his shoulder", "looking over her shoulder",
                              "looking at his paper", "bubbling in", "bubbling answers"],
            "key_dialogue":  ["nothing is going on", "don't tell on me", "shhh", "shhhh",
                              "said nothing", "kept quiet", "stayed quiet", "looked away",
                              "nothing going on", "continued the test", "shh"],
            "relationship":  ["best friend", "since they were toddlers",
                              "whisper", "whispered"],
        },
        "global": {
            "honesty":        ["honest", "honesty", "truth", "truthful",
                               "lie", "lying", "integrity",
                               "academic dishonesty", "academic integrity"],
            "loyalty":        ["friend", "friendship", "loyalty", "loyal",
                               "trust", "trusted", "betray", "betrayal"],
            "moral_conflict": ["right", "wrong", "moral", "ethical", "conscience",
                               "dilemma", "conflict", "torn", "hard decision",
                               "caught between", "conflicted", "moral dilemma",
                               "difficult decision", "difficult choice"],
            "social_code":    ["snitch", "tell on", "protect", "cover",
                               "secret", "keeping quiet", "staying silent", "report",
                               "witness", "witnessed", "saw it happen",
                               "speak up", "spoke up", "cover up", "covered for",
                               "covering for", "say something", "tell someone"],
            "consequences":   ["consequence", "get in trouble", "punishment",
                               "get caught", "cheating has consequences"],
        },
        "causal": {
            "connectors":  ["because", "that's why", "which is why", "since",
                            "as a result", "therefore", "thus",
                            "even though", "I know that"],
            "motivation":  ["didn't want", "to protect", "to keep", "to avoid",
                            "chose not to", "decided not to", "afraid", "worried", "scared"],
            "explanation": ["this shows", "this means", "which means",
                            "so she", "explains why", "that is why",
                            "in the comic", "in the text", "in the story"],
        },
        "themes": {
            "loyalty_vs_honesty": ["loyal", "honest", "conflict", "dilemma",
                                   "torn", "caught between", "hard decision",
                                   "trade-off", "on one hand"],
            "friendship":         ["friend", "best friend", "trust", "bond",
                                   "close", "knew each other", "their friendship",
                                   "didn't want to lose"],
            "integrity":          ["integrity", "right thing", "wrong thing",
                                   "moral", "should have", "guilty", "guilt",
                                   "knew it was wrong", "regret"],
            "peer_loyalty":       ["snitch", "tell on", "protect", "cover",
                                   "secret", "code of silence", "keep quiet",
                                   "friends don't tell"],
            "fairness":           ["fair", "unfair", "fairness", "equal",
                                   "others studied", "others worked hard", "advantage",
                                   "not fair", "deserve"],
            "academic_integrity": ["academic dishonesty", "academic integrity",
                                   "cheating is wrong", "not allowed to cheat",
                                   "against the rules"],
        },
    },

    "stealing": {
        "local": {
            "characters":       ["billy"],
            "objects":          ["bike", "bicycle", "red bike", "beautiful bike", "wallet"],
            "money":            ["300", "$300", "three hundred", "dollars"],
            "setting":          ["beach", "table", "ice cream shop", "birthday", "outside",
                                 "for his birthday"],
            "parents_context":  ["parents said no", "mom and dad", "mom", "dad", "said no",
                                 "birthday present", "couldn't get", "wouldn't buy"],
            "key_moments":      ["forgot her wallet", "grabbed the wallet", "opened the wallet",
                                 "no i don't", "can't find", "looking for her wallet",
                                 "lied", "told her no", "said he didn't know",
                                 "came back looking", "came back"],
        },
        "global": {
            "stealing":     ["steal", "stealing", "theft", "thief",
                             "wrong", "bad", "dishonest", "dishonesty"],
            "moral":        ["moral", "conscience", "guilty", "guilt",
                             "regret", "lie", "lying", "lied",
                             "lied about it", "denied", "deny"],
            "temptation":   ["temptation", "tempted", "resist", "couldn't resist",
                             "opportunity", "took advantage", "desire",
                             "greed", "greedy", "gave in",
                             "wants vs needs", "want vs need", "needs and wants"],
            "consequences": ["consequence", "result", "trouble", "punishment",
                             "karma", "get caught", "face the consequences",
                             "wrong choice", "bad choice", "bad decision"],
        },
        "causal": {
            "connectors":  ["because", "that's why", "which is why", "since",
                            "as a result", "therefore", "thus",
                            "even though", "I know that"],
            "motivation":  ["wanted to", "needed to", "to get", "decided to",
                            "chose to", "couldn't resist", "led him"],
            "explanation": ["tempted him", "made him", "so he",
                            "felt that", "this shows", "explains why",
                            "in the comic", "in the text", "in the story"],
        },
        "themes": {
            "temptation":     ["tempt", "temptation", "couldn't resist", "gave in",
                               "opportunity", "easy", "saw the chance",
                               "took advantage", "just sitting there"],
            "honesty":        ["honest", "honesty", "lie", "lying", "dishonest",
                               "truth", "admit", "confession", "should have told",
                               "came clean"],
            "consequences":   ["consequence", "result", "regret", "guilt", "guilty",
                               "punishment", "wrong", "face the", "will regret",
                               "end up"],
            "greed":          ["greed", "greedy", "selfish", "want more",
                               "envy", "jealous", "desire", "not satisfied",
                               "wanted more", "couldn't afford"],
            "privilege":      ["privilege", "privileged", "rich", "spoiled", "spoil",
                               "earn", "deserve", "entitled", "poor", "didn't earn it"],
            "wants_vs_needs": ["wants vs needs", "want vs need", "needs and wants",
                               "didn't need it", "wanted not needed", "just wanted"],
        },
    },
}

# ── Exemplars ──────────────────────────────────────────────────────────────────
# Two exemplar responses per (story × waypoint). Each exemplar covers all 4
# questions. Exemplars are embedded and cached at startup; responses are scored
# by cosine similarity to them.
#
# Question mapping (same for all stories):
#   q1 = {story}_1  — inference        (1a: specific why/how question)
#   q2 = {story}_2  — metacognitive    (1b: "what made you think of that?")
#   q3 = {story}_3  — inference        (2a: lesson question)
#   q4 = {story}_4  — metacognitive    (2b: "what made you think of that?")

EXEMPLARS = {
    "cat": {
        "WAYPOINT0": [
            {
                "q1": "I don't know why he kept it.",
                "q2": "I just don't know.",
                "q3": "I don't really know what the lesson is.",
                "q4": "I couldn't think of anything.",
            },
            {
                "q1": "Because cats are fluffy.",
                "q2": "My cat at home is fluffy.",
                "q3": "Cats need food and water.",
                "q4": "That's what my mom always says.",
            },
        ],
        "WAYPOINT1": [
            {
                "q1": "He kept the cat because it was sitting in the rain, all wet and cold and shaking, and he felt bad for it.",
                "q2": "The story said he saw how sad the cat was and bent down and petted it even when his friends told him not to.",
                "q3": "The lesson is that you should be nice to animals that are cold and wet and need help.",
                "q4": "Because the boys in the story were being mean to the cat but Bruno still took it home.",
            },
            {
                "q1": "The boy kept the cat because it was out in the rain getting cold and none of the other boys would help it.",
                "q2": "It said in the story that the boys were laughing at the cat and Bruno walked over and petted it anyway.",
                "q3": "The lesson is you should take care of stray animals if they are wet and cold and nobody wants them.",
                "q4": "Because that's what happened — the cat was shaking in the rain and Bruno gave it a new home.",
            },
        ],
        "WAYPOINT2": [
            {
                "q1": "He kept the cat because true friends look out for each other and don't care what other people think.",
                "q2": "That's just how real friendship works — you stand up for who you care about even if it's embarrassing.",
                "q3": "The lesson is that peer pressure can stop you from doing the right thing, but it takes courage to be kind anyway.",
                "q4": "I've seen people get made fun of for being nice and it's hard but the right thing is to be kind regardless.",
            },
            {
                "q1": "He kept the cat because being a good person means showing empathy even when others judge you for it.",
                "q2": "I know from life that if you only do good things when it's easy, it doesn't really mean much.",
                "q3": "The lesson is about not giving in to peer pressure. Sometimes doing the right thing means going against the group.",
                "q4": "I've seen this happen — people don't stand up for others because they're scared of being left out.",
            },
        ],
        "WAYPOINT3": [
            {
                "q1": "The red headed boy kept the cat because he realized that the cat did not deserve to be bullied by the other students.",
                "q2": "Friends should not be bullied, and if they are being bullied, we should be there to support them.",
                "q3": "One lesson that can be learned is to always be there for your friends and to have empathy for those who are bullied.",
                "q4": "It feels like the right thing to do, and it is what I am taught. The boy also did something nice for the cat, which implies empathy.",
            },
            {
                "q1": "The boy kept the cat because he understood that the cat was a true friend, better than the bullies from before.",
                "q2": "The boy realized that the bullies were not good friends, and that the cat was a better friend from the start.",
                "q3": "We should not judge other people and make fun of them.",
                "q4": "It is a mean thing to judge others and make people feel like they don't belong.",
            },
            {
                "q1": "He kept the cat because he felt bad for his friends bullying the cat and wanted to thank the cat for comforting him. Everyone should be welcomed and accepted without being commented on their appearances.",
                "q2": "I thought of this answer because the cat was being called stinky after it comforted the boy. I saw from the comic that the cat started crying and no one should be left aside and made fun of.",
                "q3": "A lesson someone can learn from this is to not bully someone, especially without knowing the backstory. The friends didn't know the cat comforted the boy and started calling the cat stinky.",
                "q4": "I thought of this answer because the cat was crying when the boys were bullying it. No one should be bullied because of their looks and left alone.",
            },
            {
                "q1": "He kept the cat because he felt bad for the cat and because his friends were bullying the cat. He didn't want the cat to be bullied since no one deserves to be treated miserably.",
                "q2": "The boy saw the cat crying, and then realized that his feelings were being hurt. The right thing to do is to help this cat since he is all alone.",
                "q3": "A lesson someone can learn is be friendly to everyone and not judge someone based on their looks, which corresponds to the boys calling the cat dirty in the story.",
                "q4": "I thought of this answer because the boys didn't even want to play with the red headed boy after he touched the cat. The cat was almost gonna be left aside. We should always take care of the people around us.",
            },
            {
                "q1": "Because the cat was nice to Bruno when he was crying on the bench and so he must have wanted to return the favor. Additionally, Bruno didn't like other boys were teasing the cat for being dirty and smelly. He took the courage not to join such a mean behavior.",
                "q2": "Because of the change in Bruno's behavior; initially he was curious about the cat, then he encounters other boys being mean, following them out of not being bullied himself, but in the end, he takes the courage to stand against the other boys.",
                "q3": "One can take the courage to stand against a mean behavior and show kindness, especially when kindness is shown towards them.",
                "q4": "Bruno changing his stance towards the cat: initially curious, followed by walking away so as not to be bullied, but then deciding to take the cat as he thought that's the right thing to do.",
            },
        ],
    },

    "lying": {
        "WAYPOINT0": [
            {
                "q1": "I don't know why she didn't tell.",
                "q2": "I just couldn't think of anything.",
                "q3": "I don't know the lesson.",
                "q4": "I don't know.",
            },
            {
                "q1": "Because she likes her desk.",
                "q2": "I like desks too.",
                "q3": "Math is hard.",
                "q4": "Math class is really long.",
            },
        ],
        "WAYPOINT1": [
            {
                "q1": "Sarah didn't tell because Rachel whispered to her and said 'Don't tell on me, I'm your best friend.'",
                "q2": "Because it says in the story that Rachel told Sarah to be quiet and reminded her they were best friends.",
                "q3": "The lesson is that you should study for your test so you don't have to cheat like Rachel did.",
                "q4": "Because Rachel didn't study but Sarah did, and Rachel had to look at someone else's answers.",
            },
            {
                "q1": "She didn't tell because the teacher asked if something was going on and Sarah said nothing was going on.",
                "q2": "The story says the teacher asked Sarah and she answered that nothing was going on.",
                "q3": "The lesson is to not cheat on tests because you might get caught and get in trouble.",
                "q4": "Because Rachel was copying and Sarah saw it and the teacher noticed Sarah looking.",
            },
        ],
        "WAYPOINT2": [
            {
                "q1": "Sarah didn't tell because loyalty to your best friend is powerful — you don't want to be the one who gets them in trouble, even when they're doing something wrong.",
                "q2": "That's a really common situation — when your friend does something wrong it's hard to turn them in because you care about them.",
                "q3": "The lesson is that honesty and loyalty can conflict, and sometimes doing the right thing means going against someone you care about.",
                "q4": "I've seen people stay quiet to protect their friends even when they knew it was wrong, because friendship loyalty feels really strong in the moment.",
            },
            {
                "q1": "She stayed quiet to protect her friend. That's what loyalty looks like — you cover for the people close to you even if it means you're not being fully honest.",
                "q2": "People generally don't want to get their friends in trouble, especially a best friend they've known since they were little.",
                "q3": "The lesson is that protecting a friend isn't always the right thing. Sometimes being a true friend means telling the truth even when it's hard.",
                "q4": "I know that telling on your friend feels like a betrayal, but there's a difference between loyalty and letting someone keep doing something wrong.",
            },
        ],
        "WAYPOINT3": [
            {
                "q1": "Sarah does not tell the teacher to protect her from getting in trouble because she does not want her friend to be in trouble.",
                "q2": "Loyal friends will try to keep each other safe and out of trouble, even if it means bending the rules a little bit.",
                "q3": "We should try to study harder so that we don't need to cheat every time. But if we fall short, we should not punish our friends.",
                "q4": "This is so that we can avoid this situation entirely by not needing to cheat, and if we get into a sticky situation, we know that our friends still have our back.",
            },
            {
                "q1": "Sarah does not want to betray her friend just because she cheated on her test.",
                "q2": "It is not nice to tell on your friends. I would not want that to happen to me if I cheated.",
                "q3": "People should not cheat because it is not a good thing to do.",
                "q4": "Cheating is not something to be proud of because it is unfair and can hurt our learning, especially if we are cheating on tests.",
            },
            {
                "q1": "She didn't tell the teacher because Rachel told her to not tell the teacher since they are best friends. Friends are suppose to help each other.",
                "q2": "I thought of that answer because friends are suppose to defend each other, especially when Rachel asked Sarah to cover for her.",
                "q3": "A lesson someone can learn is that lying to defend your friend can be unfair to others. The boy in the picture didn't know at all his answer was being copied.",
                "q4": "I thought of this answer because it is unfair to have your answer copied. In the comic, Sarah lied to the teacher even though she knew that it would be unfair to the boy and to everyone else in the class.",
            },
            {
                "q1": "She didn't tell the teacher because Rachel is her best friend. We often feel the need to defend people we feel closest to us and to not get them in trouble.",
                "q2": "I thought of that answer because Rachel begged her to not tell the teacher. Sarah didn't want to tell because best friends don't betray each other.",
                "q3": "A lesson someone can learn is that sometimes defending people you feel close to isn't the best way to guide them to do the right thing. Sarah didn't snitch on Rachel, which is unfair to the student that got his answer copied.",
                "q4": "I thought of this answer because Sarah lied to the teacher and lying isn't the best way to approach this kind of situation. It is unfair to everyone else in the class.",
            },
            {
                "q1": "Because Rachel is her best friend, and we help our friends when we can.",
                "q2": "Friendship overweights morality sometimes; plus Rachel explicitly tells Sarah not to tell the teacher.",
                "q3": "Not sure if there is a lesson. Sarah appears to be weighing her friendship to Rachel more than being morally correct.",
                "q4": "Because she basically followed what Rachel told her to do and didn't confront her about cheating.",
            },
        ],
    },

    "stealing": {
        "WAYPOINT0": [
            {
                "q1": "I don't know.",
                "q2": "I just don't know.",
                "q3": "I don't really know the lesson.",
                "q4": "I can't think of anything.",
            },
            {
                "q1": "Because he wanted ice cream.",
                "q2": "Ice cream is good.",
                "q3": "Ice cream makes you feel better.",
                "q4": "Because the story talked about ice cream.",
            },
        ],
        "WAYPOINT1": [
            {
                "q1": "Billy kept the wallet because he saw there was $300 in it and wanted to use it to buy the red bike he had seen.",
                "q2": "The story said Billy opened the wallet and thought 'With that money I could buy that red bike!'",
                "q3": "The lesson is that you shouldn't take things that don't belong to you, because it's stealing.",
                "q4": "Because Billy took the lady's wallet and then lied to her when she came back looking for it.",
            },
            {
                "q1": "He kept it because his parents had already said no to the bike and now he saw $300 just sitting in the wallet.",
                "q2": "The story showed Billy had already been told no for the bike, so when he found the wallet with exactly $300 he decided to keep it.",
                "q3": "The lesson is that you shouldn't steal someone's wallet and lie about it. Billy should have given it back.",
                "q4": "Because Billy told the lady he hadn't seen her wallet even though he had it the whole time.",
            },
        ],
        "WAYPOINT2": [
            {
                "q1": "Billy kept the wallet because greed and desire clouded his judgment. He wanted the bike so badly that when temptation appeared, he gave in instead of doing the right thing.",
                "q2": "People often make bad decisions when they want something really badly and an easy opportunity shows up. That's just human nature — temptation is hard to resist.",
                "q3": "The lesson is that greed and dishonesty don't pay. Even if you get what you want in the short term, doing wrong always has consequences.",
                "q4": "I've seen people make bad choices when they really want something. Temptation is powerful and it's easy to rationalize taking something if it seems like an easy chance.",
            },
            {
                "q1": "He kept it because the temptation was too strong. He wanted that bike badly, his parents said no, and suddenly $300 just appeared. Desire and frustration made it easy to make the wrong choice.",
                "q2": "That's a classic temptation story — when you want something you can't have and an opportunity lands in front of you, it's very hard to do the right thing.",
                "q3": "The lesson is about resisting temptation and being honest. Taking what isn't yours might give you what you want for now, but you have to live with the dishonesty.",
                "q4": "Greed is a really powerful feeling. Most people know stealing is wrong but temptation makes you rationalize it, especially when you feel like you're missing out.",
            },
        ],
        "WAYPOINT3": [
            {
                "q1": "Billy kept the lady's wallet because he wanted to keep the money to buy a new bike.",
                "q2": "Billy was upset that his parents won't buy him a new bike, so when he found the money, he wanted to use it for the new bike.",
                "q3": "One lesson could be to always double check before leaving so that you don't forget anything.",
                "q4": "I would not want to lose something and have someone steal it. If I didn't forget anything, people would not have the chance to steal from me.",
            },
            {
                "q1": "Billy kept the lady's wallet because he thought about using the money inside to buy a new bike.",
                "q2": "Billy expressed interest in a new bike, but he could not get it because it was too expensive.",
                "q3": "Don't steal from other people, but instead work to earn the money some other way to get what you want.",
                "q4": "I have been taught that if I want something, I should be on my best behavior, or do something to earn it. Nothing comes for free.",
            },
            {
                "q1": "Billy kept the wallet because he really wanted the red bike. Having that would make him look as cool as the others, and would make him happier.",
                "q2": "I thought of that answer because of the way Billy was looking at the bike, and knowing that free money can bring people a lot of joy.",
                "q3": "A lesson someone can learn is that lying to someone about not having something that belongs to them is the same as stealing. In the story, it is clear that Billy stole her wallet since he didn't want to give it back.",
                "q4": "I thought of the answer from seeing how sad that lady was when she asked him. I know that if you don't return something back to their owner, this action counts as stealing.",
            },
            {
                "q1": "Billy kept the wallet because he wanted the money for the bike. He thinks that if he lies, the lady will never find out that he stole the money.",
                "q2": "I thought of that answer because Billy thought to himself that he can use the 300 to buy a new bike. This way he doesn't have to get his parents' approval.",
                "q3": "A lesson someone can learn is that you shouldn't steal something in the goal of benefitting yourself. It could bring harm to others, such as the lady losing 300 dollars.",
                "q4": "I thought of that answer from seeing Billy's reasoning from the text and inferring how sad the lady will be after hearing his answers.",
            },
            {
                "q1": "Because he wanted to buy the new expensive bike.",
                "q2": "His parents indicated that they won't buy him the bike, so he needed to find another way to buy it.",
                "q3": "One should not keep other's money just because they want to buy something.",
                "q4": "Because it is not morally correct to lie about the lady's wallet when he took it.",
            },
        ],
    },
}

# ── Exemplar embedding cache ───────────────────────────────────────────────────

_exemplar_cache: dict = {}   # (story, waypoint, ex_idx, q_key) -> np.ndarray


def _build_exemplar_cache():
    if EMBEDDING_BACKEND is None:
        return

    texts, keys = [], []
    for story in EXEMPLARS:
        for waypoint in EXEMPLARS[story]:
            for ex_idx, ex in enumerate(EXEMPLARS[story][waypoint]):
                for q_key, q_text in ex.items():
                    k = (story, waypoint, ex_idx, q_key)
                    if k not in _exemplar_cache:
                        texts.append(q_text)
                        keys.append(k)

    if not texts:
        return

    n = len(texts)
    print(f"Pre-computing {n} exemplar embeddings...")
    embeddings = embed_texts(texts)
    for k, emb in zip(keys, embeddings):
        _exemplar_cache[k] = emb

    # Warn if any embeddings failed — silent 0.0 scores are worse than a clear error
    succeeded = sum(1 for v in _exemplar_cache.values() if v is not None)
    if succeeded < n:
        print(f"WARNING: Only {succeeded}/{n} exemplar embeddings succeeded.")
        print("         Semantic scores for uncached exemplars will be 0.0, not real similarities.")
        print("         Check your embedding backend and authentication.\n")
    else:
        print(f"Exemplar cache ready ({succeeded} vectors).\n")


# ── Keyword matching utilities ─────────────────────────────────────────────────

def _lemmatize(word):
    return LEMMATIZER.lemmatize(word.lower()) if NLTK_AVAILABLE else word.lower()


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
    # Window extended to 5 intervening words to catch constructions like
    # "he was not particularly kind" or "she did not really stand up".
    pattern = (
        r'\b(?:' + '|'.join(re.escape(n) for n in NEGATION_WORDS) + r')'
        r'(?:\s+\w+){0,5}\s+' + re.escape(keyword.split()[0])
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def _word_in_text(word, text_lower):
    """Word-boundary-aware lookup. Phrases use substring; single words use \\b to
    prevent 'cat' matching 'catalog' or 'indicate'."""
    if " " in word:
        return word in text_lower
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text_lower))


def _keyword_hit(text_lower, tokens, lemmas, kw):
    """4-tier match for one keyword. Returns True if matched and not negated."""
    kw_lower = kw.lower()

    # Tier 1: exact match with word boundaries for single words
    if _word_in_text(kw_lower, text_lower):
        return not _is_negated(text_lower, kw_lower)

    # Tier 2: lemmatized token match
    kw_lemma = _lemmatize(kw_lower.split()[0])
    if kw_lemma in lemmas:
        return not _is_negated(text_lower, kw_lower)

    # Tier 3: WordNet synonym expansion (also word-boundary-aware)
    if NLTK_AVAILABLE:
        for syn in _get_synonyms(kw_lower.split()[0]):
            if _word_in_text(syn, text_lower) or _lemmatize(syn.split()[0]) in lemmas:
                return not _is_negated(text_lower, syn)

    # Tier 4: fuzzy token match (single words >= 5 chars only)
    if RAPIDFUZZ_AVAILABLE and len(kw_lower) >= 5 and " " not in kw_lower:
        for tok in tokens:
            if len(tok) >= 4 and fuzz.ratio(kw_lower, tok) >= FUZZY_THRESHOLD:
                return not _is_negated(text_lower, kw_lower)

    return False


def _list_hit(text_lower, tokens, lemmas, keywords):
    """True if any keyword from a flat list matches via the 4-tier pipeline."""
    return any(_keyword_hit(text_lower, tokens, lemmas, kw) for kw in keywords)


def cluster_score(text, cluster_dict):
    """clusters_hit / total_clusters. A cluster is hit if any keyword passes the 4-tier pipeline."""
    if not isinstance(text, str) or not text.strip() or not cluster_dict:
        return 0.0
    text_lower = text.lower()
    tokens = _tokenize(text_lower)
    lemmas = {_lemmatize(t) for t in tokens}
    hits = 0
    for keywords in cluster_dict.values():
        if _list_hit(text_lower, tokens, lemmas, keywords):
            hits += 1
    return hits / len(cluster_dict)


def causal_cluster_score(text, causal_dict, local_dict, global_dict):
    """Causal score weighted by the average of local and global grounding.

    Using the mean instead of max means a response must show BOTH local and
    global content to get full grounding credit. This separates WAYPOINT3
    (high local + high global → grounding near 1.0) from WAYPOINT2
    (high global but near-zero local → grounding ~0.5 at best), which the
    old max() formula could not do.
    """
    local_s  = cluster_score(text, local_dict)
    global_s = cluster_score(text, global_dict)
    grounding = (local_s + global_s) / 2
    if grounding == 0.0:
        return 0.0
    return cluster_score(text, causal_dict) * grounding


# ── Semantic scoring ───────────────────────────────────────────────────────────

def _semantic_from_embedding(resp_emb, story, q_key):
    """
    Given a pre-computed response embedding, return similarity scores against
    all exemplars for each waypoint.
    """
    empty = {
        "sim_WAYPOINT0": np.nan, "sim_WAYPOINT1": np.nan,
        "sim_WAYPOINT2": np.nan, "sim_WAYPOINT3": np.nan,
        "prediction_top_sim": np.nan, "prediction_margin": np.nan,
        "predicted_waypoint": None, "semantic_quality": np.nan,
    }
    if resp_emb is None:
        return empty

    sim_by_wp = {}
    for wp in WAYPOINTS:
        n_ex = len(EXEMPLARS.get(story, {}).get(wp, []))
        sims = [
            cosine_sim(resp_emb, _exemplar_cache.get((story, wp, i, q_key)))
            for i in range(n_ex)
        ]
        sims = [s for s in sims if s is not None]
        sim_by_wp[wp] = max(sims) if sims else 0.0

    sorted_sims = sorted(sim_by_wp.values(), reverse=True)
    top_sim    = sorted_sims[0]
    second_sim = sorted_sims[1] if len(sorted_sims) > 1 else 0.0
    margin     = top_sim - second_sim

    # Only assign a prediction when the top score is high enough AND
    # it clearly beats the second-best level. Near-ties stay as None.
    if top_sim >= PRED_MIN_SIM and margin >= PRED_MIN_MARGIN:
        predicted = max(sim_by_wp, key=sim_by_wp.get)
    else:
        predicted = None

    sum_weights = sum(SEM_WEIGHTS.values())  # 0 + 1/3 + 2/3 + 1.0 = 2.0
    semantic_quality = sum(sim_by_wp[wp] * SEM_WEIGHTS[wp] for wp in WAYPOINTS) / sum_weights

    return {
        "sim_WAYPOINT0":      round(sim_by_wp["WAYPOINT0"], 4),
        "sim_WAYPOINT1":      round(sim_by_wp["WAYPOINT1"], 4),
        "sim_WAYPOINT2":      round(sim_by_wp["WAYPOINT2"], 4),
        "sim_WAYPOINT3":      round(sim_by_wp["WAYPOINT3"], 4),
        "prediction_top_sim": round(top_sim, 4),
        "prediction_margin":  round(margin, 4),
        "predicted_waypoint": predicted,
        "semantic_quality":   round(semantic_quality, 4),
    }


# ── Per-question analysis ──────────────────────────────────────────────────────

QUESTION_TYPE = {"q1": "inference", "q2": "metacognitive",
                 "q3": "inference",  "q4": "metacognitive"}

STORY_COLS = {
    "cat":      ["cat_1",      "cat_2",      "cat_3",      "cat_4"],
    "lying":    ["lying_1",    "lying_2",    "lying_3",    "lying_4"],
    "stealing": ["stealing_1", "stealing_2", "stealing_3", "stealing_4"],
}
Q_KEYS = ["q1", "q2", "q3", "q4"]


def _keyword_scores(text, story):
    banks = WORD_BANKS[story]
    local_s  = cluster_score(text, banks["local"])
    global_s = cluster_score(text, banks["global"])
    causal_s = causal_cluster_score(text, banks["causal"], banks["local"], banks["global"])
    # themes: {theme_name: [keywords]} — each theme is a single cluster (flat list)
    _tl = text.lower() if isinstance(text, str) else ""
    _tk = _tokenize(_tl)
    _lm = {_lemmatize(t) for t in _tk}
    theme_scores = {
        name: (1.0 if _list_hit(_tl, _tk, _lm, keywords) else 0.0)
        for name, keywords in banks["themes"].items()
    }
    # Derive mean from already-computed theme_scores to avoid a redundant pass
    mean_themes = sum(theme_scores.values()) / len(theme_scores) if theme_scores else 0.0
    kw_composite = (W_LOCAL * local_s + W_GLOBAL * global_s +
                    W_CAUSAL * causal_s + W_THEMES * mean_themes)
    return {
        "anchor_local":       round(local_s, 4),
        "anchor_global":      round(global_s, 4),
        "anchor_causal":      round(causal_s, 4),
        "anchor_mean_themes": round(mean_themes, 4),
        **{f"anchor_theme_{k}": round(v, 4) for k, v in theme_scores.items()},
        "keyword_composite":  round(kw_composite, 4),
    }


def build_long_df(df):
    """
    Two-pass construction:
      Pass 1 — keyword scores (no API calls).
      Pass 2 — batch embed all response texts, then compute semantic scores.
    Returns one row per (respondent × story × question).
    """
    rows = []
    all_texts = []   # parallel list to rows

    for _, r in df.iterrows():
        for story, cols in STORY_COLS.items():
            for col, q_key in zip(cols, Q_KEYS):
                text = str(r[col]) if col in r.index and pd.notna(r[col]) else ""
                kw = _keyword_scores(text, story)
                rows.append({
                    "respondent_id":        r.get("respondent_id", ""),
                    "level":                r.get("profile_level", ""),
                    "format":               r.get("format", ""),
                    "experience":           r.get("experience", ""),
                    "story":                story,
                    "question":             q_key,
                    "question_type":        QUESTION_TYPE[q_key],
                    "response_text":        text,
                    "response_length_words": len(text.split()) if text else 0,
                    **kw,
                })
                all_texts.append((text, story, q_key))

    print(f"Keyword scoring complete ({len(rows)} rows).")

    # Safety check: both lists must be in lockstep before the embedding pass
    assert len(rows) == len(all_texts), (
        f"Internal sync error: rows={len(rows)}, all_texts={len(all_texts)}"
    )

    # Pass 2: embed all responses in one batch call
    if EMBEDDING_BACKEND is not None:
        raw_texts = [t for t, _, _ in all_texts]
        print(f"Embedding {len(raw_texts)} responses (backend: {EMBEDDING_BACKEND})...")
        embeddings = embed_texts(raw_texts)
        print("Computing semantic scores...")
        for i, ((text, story, q_key), emb) in enumerate(zip(all_texts, embeddings)):
            sem = _semantic_from_embedding(emb, story, q_key)
            rows[i].update(sem)
            kw_comp = rows[i]["keyword_composite"]
            sq = sem.get("semantic_quality")
            scored = pd.notna(sq)
            rows[i]["has_semantic_score"] = scored
            if scored:
                rows[i]["overall_composite"] = round(W_KEYWORD * kw_comp + W_SEMANTIC * sq, 4)
            else:
                rows[i]["overall_composite"] = kw_comp
    else:
        for row in rows:
            row.update({
                "sim_WAYPOINT0": np.nan, "sim_WAYPOINT1": np.nan,
                "sim_WAYPOINT2": np.nan, "sim_WAYPOINT3": np.nan,
                "prediction_top_sim": np.nan, "prediction_margin": np.nan,
                "predicted_waypoint": None, "semantic_quality": np.nan,
                "has_semantic_score": False,
                "overall_composite": row["keyword_composite"],
            })

    return pd.DataFrame(rows)


# ── Aggregation ────────────────────────────────────────────────────────────────

SCORE_COLS = [
    "anchor_local", "anchor_global", "anchor_causal", "anchor_mean_themes",
    "keyword_composite", "semantic_quality", "overall_composite",
]


def story_level_summary(long_df):
    return long_df.groupby(["level", "story"])[SCORE_COLS].mean(numeric_only=False).round(3)


def qtype_summary(long_df):
    return long_df.groupby(["level", "story", "question_type"])[SCORE_COLS].mean(numeric_only=False).round(3)


def per_question_summary(long_df):
    return long_df.groupby(["level", "story", "question"])[SCORE_COLS].mean(numeric_only=False).round(3)


def waypoint_accuracy(long_df):
    """
    Returns (accuracy, coverage) — two DataFrames, both level × story.

    accuracy : fraction of confident predictions that matched the assigned level.
               Only includes responses that cleared the confidence threshold.
    coverage : fraction of responses that received ANY confident prediction.
               Low coverage means the thresholds are too strict or embeddings failed.
    """
    if "predicted_waypoint" not in long_df.columns or long_df["predicted_waypoint"].isna().all():
        return pd.DataFrame(), pd.DataFrame()

    coverage = (
        long_df.groupby(["level", "story"])["predicted_waypoint"]
        .apply(lambda x: x.notna().mean())
        .round(3)
        .unstack("story")
    )

    sub = long_df.dropna(subset=["predicted_waypoint"]).copy()
    sub["correct"] = sub["predicted_waypoint"] == sub["level"]
    accuracy = sub.groupby(["level", "story"])["correct"].mean().round(3).unstack("story")

    return accuracy, coverage


def distinctiveness_diagnostic(summary):
    EXPECTED = {
        "anchor_local":      ["WAYPOINT1", "WAYPOINT3", "WAYPOINT2", "WAYPOINT0"],
        "anchor_global":     ["WAYPOINT2", "WAYPOINT3", "WAYPOINT1", "WAYPOINT0"],
        "anchor_causal":     ["WAYPOINT3", "WAYPOINT2", "WAYPOINT1", "WAYPOINT0"],
        "overall_composite": ["WAYPOINT3", "WAYPOINT2", "WAYPOINT1", "WAYPOINT0"],
    }
    violations = []
    for story in ["cat", "lying", "stealing"]:
        try:
            sd = summary.xs(story, level="story")
        except KeyError:
            continue
        for anchor, order in EXPECTED.items():
            if anchor not in sd.columns:
                continue
            avail = [lvl for lvl in order if lvl in sd.index]
            if len(avail) < 2:
                continue
            scores = sd.loc[avail, anchor]
            for i in range(len(avail) - 1):
                hi, lo = avail[i], avail[i + 1]
                if scores[hi] < scores[lo]:
                    violations.append({
                        "story": story, "anchor": anchor,
                        "violation": f"{hi} ({scores[hi]:.3f}) < {lo} ({scores[lo]:.3f})",
                        "expected": f"{hi} > {lo}",
                    })
    return pd.DataFrame(violations)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(input_csv=DEFAULT_INPUT, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        print(f"       Usage: python hybrid_anchor_analysis.py path/to/responses.csv")
        return

    _init_embedding_backend()
    _build_exemplar_cache()

    df = pd.read_csv(input_csv)

    # Backwards-compatibility: convert IIR0-3 labels to WAYPOINT0-3
    if "profile_level" in df.columns:
        df["profile_level"] = df["profile_level"].str.replace(
            r"^IIR(\d)$", r"WAYPOINT\1", regex=True
        )

    print(f"\nLoaded {len(df)} respondents from {input_csv}")
    print(f"Level breakdown:\n{df['profile_level'].value_counts().sort_index().to_string()}\n")

    print("Running hybrid anchor analysis...")
    long_df = build_long_df(df)
    print("Done.\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []

    def _save(table, name):
        path = os.path.join(output_dir, f"{name}_{ts}.csv")
        table.to_csv(path)
        saved.append(path)
        return path

    # every question response with its full keyword + semantic scores
    detail_path = os.path.join(output_dir, f"scores_every_response_{ts}.csv")
    long_df.to_csv(detail_path, index=False)
    saved.append(detail_path)

    # mean scores grouped by waypoint level and story
    story_sum = story_level_summary(long_df)
    _save(story_sum, "summary_by_waypoint_and_story")

    # mean scores split by inference questions (1a, 2a) vs metacognitive questions (1b, 2b)
    qt_sum = qtype_summary(long_df)
    _save(qt_sum, "summary_by_question_type_inference_vs_metacognitive")

    # mean scores for each individual question q1 through q4
    pq_sum = per_question_summary(long_df)
    _save(pq_sum, "summary_by_individual_question_q1_to_q4")

    acc, cov = waypoint_accuracy(long_df)
    if not acc.empty:
        # fraction of confident predictions that matched the assigned level
        _save(acc, "semantic_waypoint_prediction_accuracy")
        # fraction of responses that received a confident prediction (coverage)
        _save(cov, "semantic_waypoint_prediction_coverage")

    # ── Print summaries ────────────────────────────────────────────────────────
    print("=" * 70)
    print("STORY-LEVEL SUMMARY  —  Mean scores by Waypoint × Story")
    print("=" * 70)
    print(story_sum.to_string(), "\n")

    print("=" * 70)
    print("QUESTION TYPE  —  Inference vs Metacognitive")
    print("=" * 70)
    print(qt_sum.to_string(), "\n")

    print("=" * 70)
    print("PER-QUESTION  —  q1 through q4")
    print("=" * 70)
    print(pq_sum.to_string(), "\n")

    if not acc.empty:
        print("=" * 70)
        print("SEMANTIC WAYPOINT PREDICTION ACCURACY")
        print("Fraction of confident predictions that matched the assigned level")
        print("(only rows that cleared PRED_MIN_SIM and PRED_MIN_MARGIN are included)")
        print("=" * 70)
        print(acc.to_string(), "\n")
        print("=" * 70)
        print("SEMANTIC WAYPOINT PREDICTION COVERAGE")
        print("Fraction of responses that received a confident prediction")
        print("(low values = thresholds too strict, or embedding backend is failing)")
        print("=" * 70)
        print(cov.to_string(), "\n")

    print("=" * 70)
    print("DISTINCTIVENESS DIAGNOSTIC")
    print("=" * 70)
    v = distinctiveness_diagnostic(story_sum)
    if v.empty:
        print("No violations — scores follow expected theoretical ordering.\n")
    else:
        print(f"{len(v)} violation(s):\n")
        print(v.to_string(index=False), "\n")

    print("=" * 70)
    print("SCORE GUIDE")
    print("=" * 70)
    print(f"""
  keyword_composite  = {W_LOCAL}*local + {W_GLOBAL}*global + {W_CAUSAL}*causal + {W_THEMES}*themes
  semantic_quality   = cosine-sim-weighted by exemplar waypoint level  (0=W0, 1=W3)
  overall_composite  = {W_KEYWORD}*keyword + {W_SEMANTIC}*semantic

  Expected ordering:
    WAYPOINT0: Low across all scores
    WAYPOINT1: High local,  low global,  low causal
    WAYPOINT2: Low local,   high global, low causal
    WAYPOINT3: High local,  high global, high causal, high overall_composite
    """)

    print("Output files:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    main(input_csv=csv_path)
