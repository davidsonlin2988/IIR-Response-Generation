# IIR-Response-Generation

Generate **simulated student responses** to short reading-comprehension stories, for research on **Inferential Integration in Reading (IIR)** — a student's ability to combine information from a text with their own world knowledge to construct meaning.

The pipeline prompts a large language model (Google Gemini, via Vertex AI) to role-play students at controlled ability levels and writing-experience levels, then writes their answers to a CSV for downstream analysis.

---

## What this produces

A balanced synthetic dataset where each "student" is defined by three independent dimensions:

| Dimension | Values | Controls |
|-----------|--------|----------|
| **Developmental waypoint** | `WAYPOINT0`, `WAYPOINT1`, `WAYPOINT2`, `WAYPOINT3` | *Depth and type of inference* (reasoning) |
| **Format** | `comic`, `text` | Whether the student reads a comic (PDF) or plain-text version of the story |
| **Experience level** | `none`, `little`, `moderate`, `lot` | *Writing style, vocabulary, fluency* — **not** reasoning depth |

This gives **4 × 2 × 4 = 32 Student Profiles**. Each student answers **3 stories** (Cat, Lying, Stealing), with 4 questions each → **12 answers per student**.

### The four waypoints

- **WAYPOINT0 — No inference:** "I don't know" or an irrelevant/nonsensical statement.
- **WAYPOINT1 — Text-implicit:** Inference drawn *only* from what the story explicitly states; no world knowledge.
- **WAYPOINT2 — Script-implicit:** Inference drawn *primarily* from background/world knowledge; text details only implicit.
- **WAYPOINT3 — Combination:** Integrates *both* explicit text information *and* world knowledge (highest coherence).

---

## Repository structure

| Path | Purpose |
|------|---------|
| [IIR_Per_Scenario_Workflow.ipynb](IIR_Per_Scenario_Workflow.ipynb) | **Main pipeline.** Generates responses one scenario at a time (3 API calls per student). Includes design balancing, resume logic, elapsed time tracking, and CSV output. |
| [IIR.ipynb](IIR.ipynb) | Minimal connection/demo notebook — initializes Vertex AI and shows an example model call. Useful for verifying access before running the full pipeline. |
| [anchor_analysis.py](anchor_analysis.py) | **Anchor analysis script.** Validates generated responses by checking for the presence of story-specific anchors across 4 word banks (local, global, causal, themes). Outputs proportion tables by waypoint level. |
| `Cat.pdf`, `Lying.pdf`, `Stealing.pdf` | Comic-format versions of the three stories. Stored locally and sent to the model as PDF documents when `format = "comic"`. |
| [IIR_outputs/](IIR_outputs/) | Output directory. Holds the generated CSV and (during a run) a progress file for resuming. |

---

## Prerequisites

- **Python 3.10+**
- A **Google Cloud project** with the **Vertex AI API** enabled and access to the Gemini models.
- **Google Cloud CLI** installed and authenticated.

### Python packages

For the pipeline:
```powershell
pip install google-cloud-aiplatform pandas
```

For the anchor analysis script:
```powershell
pip install pandas nltk rapidfuzz
```

---

## Setup

### 1. Install the Google Cloud CLI
Download and install from: https://cloud.google.com/sdk/docs/install

### 2. Authenticate
```powershell
gcloud auth application-default login
```
This opens a browser — sign in with the Google account that has access to your GCP project.

### 3. Set your project
The pipeline reads the GCP project from the `PROJECT_CODE` environment variable. Set it permanently via **System Properties > Environment Variables**, or temporarily in PowerShell:
```powershell
$env:PROJECT_CODE = "your-gcp-project-id"
```

### 4. (Optional) Adjust the model
Defined near the top of the main notebook:
```python
MODEL_NAME = "gemini-3.1-pro-preview"
```

---

## Running the pipeline

Open [IIR_Per_Scenario_Workflow.ipynb](IIR_Per_Scenario_Workflow.ipynb) in VS Code and run the cell, or call the runner directly:

```python
run_scenario_at_a_time_pipeline(total_n=96, master_seed=53)
```

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `total_n` | `96` (notebook `__main__` uses `32`) | Number of simulated students. **Use a multiple of 32** for a perfectly balanced design. |
| `master_seed` | `53` | Seed for all assignments and shuffles — makes the entire run reproducible. |
| `level_proportions` | 25% each | Proportion of students at each waypoint. |
| `output_csv` | `IIR_outputs/IIR_TEST_1.csv` | Output path. |
| `resume` | `True` | Resume from a prior interrupted run if a matching progress file exists. |

### Recommended sample sizes (32 cells)

| `total_n` | Students per cell | Notes |
|-----------|-------------------|-------|
| 32  | 1 | Technically uniform, too thin for analysis |
| 64  | 2 | Very thin |
| 96  | 3 | Recommended minimum |
| 128 | 4 | Perfectly balanced |
| 160 | 5 | Smallest size where cell-level patterns read clearly |

### Progress & timing

While running, the pipeline prints per-student timing and an ETA:
```
[1/32] R0001 level=WAYPOINT0 format=text experience=none | elapsed=00:00:00
  -> done in 4.3s | total elapsed=00:00:04 | ETA=00:02:10 (31 students left)
```

---

## Running the anchor analysis

After generating responses, validate them with the anchor analysis script:

```powershell
python anchor_analysis.py IIR_outputs/IIR_TEST_1.csv
```

### What it checks

For each scenario (Cat, Lying, Stealing), responses are checked against 4 word banks:

| Bank | What it flags |
|------|--------------|
| **Local** | Concrete details explicitly stated in the text (names, objects, events) |
| **Global** | World knowledge / background schema |
| **Causal** | Language connecting local and global reasoning |
| **Themes** | Theme-specific banks (loyalty, empathy, peer pressure, honesty, greed, etc.) |

Matching is robust — catches exact matches, word form variants (steal/stole/stealing), synonyms, and close misspellings.

### Expected patterns

| Waypoint | Local | Global | Causal | Themes |
|----------|-------|--------|--------|--------|
| WAYPOINT0 | Low | Low | Low | Low |
| WAYPOINT1 | High | Low | Low | Low–Mod |
| WAYPOINT2 | Low | High | Low | Moderate |
| WAYPOINT3 | High | High | High | High |

### Output files

| File | Description |
|------|-------------|
| `IIR_outputs/anchor_analysis_long.csv` | Per-respondent per-story detail with all anchor flags |
| `IIR_outputs/anchor_proportions_by_level.csv` | Main summary: proportions by waypoint × story |
| `IIR_outputs/anchor_proportions_by_level_format.csv` | Same split by comic vs text |
| `IIR_outputs/anchor_proportions_by_level_experience.csv` | Same split by experience level |

---

## Output format (pipeline CSV)

The pipeline writes one row per student:

| Column | Description |
|--------|-------------|
| `respondent_id` | `R0001`, `R0002`, … |
| `profile_level` | Assigned waypoint (`WAYPOINT0`–`WAYPOINT3`) |
| `format` | `comic` or `text` |
| `experience` | `none` / `little` / `moderate` / `lot` |
| `scenario_order` | Order stories were presented, e.g. `Lying\|Cat\|Stealing` |
| `cat_1`…`cat_4` | Answers to the Cat story's 4 questions |
| `lying_1`…`lying_4` | Answers to the Lying story's 4 questions |
| `stealing_1`…`stealing_4` | Answers to the Stealing story's 4 questions |

For each story the four questions are two `(a)/(b)` pairs: an inference question and a "what made you think of that?" follow-up.

---

## How the prompting works

- **System prompt** — static and identical for every call. Establishes the instructor persona, defines the four waypoints, and notes that experience levels exist.
- **User prompt** — varies per student. Includes the assigned waypoint and experience level (with full behavioral definitions), the story (as text) or comic (as an attached PDF), the questions, and the required JSON output schema.

The experience block carries an explicit guardrail: experience affects **only** writing style/fluency, while reasoning depth is fully determined by the waypoint. (e.g. a `none`-experience `WAYPOINT3` student still integrates text and world knowledge — just in simpler language.)

---

## Reproducibility & resume

- **Deterministic:** Given the same `master_seed` and `total_n`, every assignment (waypoint, format, experience, scenario order, and question-pair shuffle) is identical across runs.
- **Resumable:** Progress is saved to `IIR_outputs/scenario_at_a_time_progress.json` after **each scenario** (not just each student), using atomic writes. An interruption loses at most one in-flight API call. On restart with matching parameters, the run picks up where it left off.
- If the saved progress parameters differ from the current run, the pipeline starts fresh and leaves the old progress file in place.

---

## Notes & caveats

- The pipeline relies on Vertex AI's **implicit** prompt caching. The system prompt may fall below caching token thresholds, in which case caching yields no savings.
- Sampling uses `temperature = 1.0` and requests `application/json` responses; malformed JSON falls back to `null` values for missing keys.
- Generation is sequential with brief sleeps between calls. A `total_n=96` run typically takes 25–75 minutes depending on model latency.
