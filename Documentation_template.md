# Business Entity Resolution — Methodology

**Team:** 404 Founders
**Final leaderboard submission:** `output_v2_s3/matching_results.tsv` — public leaderboard macro F0.5 = 0.956125

> The document follows the four required parts: **1. Methodology**, **2. Candidate generation /
> blocking strategy**, **3. Model architecture and feature engineering**, **4. Other relevant
> information** (data analysis, validation, results, error analysis, fair play, reproducibility).

---

# 1. Methodology

## 1.1 Task and metric
For every Source 1 entity in the test set we output the Source 2 / Source 3 records that describe
the same business. Scoring is F0.5 per Source 1 entity, macro-averaged, with singletons included
(an entity with no true match scores 1.0 for an empty prediction and 0.0 for any match). Precision
therefore counts twice as much as recall, and "predict nothing" is a first-class answer.

## 1.2 Pipeline overview

```
raw TSV (sep="\t", all fields as strings)
  -> [clean]   normalization.py      Indian scripts -> Latin, abbreviations, legal forms, typos,
                                      website names, postcode / state / house number / street
  -> [block]   blocking.py            11 rare-token key types (country-prefixed) + TF-IDF top-K
                                      = candidate_pairs.tsv  (exactly the pairs the model scores)
  -> [features] features.py           ~76 pair features (1 agree / 0 conflict / NaN missing)
  -> [stage 1] LightGBM                pair probability, entity-grouped out-of-fold scores
  -> [stage 2] LightGBM                + per-entity context (rank, gap, margin, other-source support)
  -> [calibrate] isotonic regression
  -> [stage 3] LightGBM on dev set     + reverse-view competition features (optional, used only if
                                      it wins on a held-back check set)
  -> [decide]  decide.py               threshold / expected-F0.5 rule + one-owner rule,
                                      tuned on a dense development set
  -> matching_results.tsv
```

## 1.3 The central idea: train and validate at test density
The test set is *dense*: every business has many look-alikes (same generic name in other cities,
same street with a different business). Our first model was trained on a *thin* sample (400k
Source 1 entities, each with its own matches and a few random other records). It scored 0.98 on a
thin holdout but **0.887** on the leaderboard, because it had rarely seen look-alikes and did not
learn to reject them.

We therefore built every training and validation step to look like the test:

1. **Dense training (`train_dense.py`)** — word rarities and name frequencies are counted over all
   training records (about the size of the test set), and each of 100,000 sampled Source 1
   entities gets its shortlist against **all** Source 2/3 training records, so its look-alikes
   are in its candidate set exactly as in the test.
2. **Dense development set (`make_devset.py`)** — the training data minus the entities used for
   training, keeping all Source 2/3 records as competitors: test density with known answers.
   The decision rule is tuned on it, and it predicts the leaderboard within about 0.01.
3. **Size-independent features** — rarity features are divided by log(N+1) and block-size limits
   scale with the data size, so the same pair gets the same features in training, dev and test.

## 1.4 Version history (what changed the score)

| Version | Change | Dev F0.5 | Leaderboard |
|---|---|---|---|
| v0 | thin training sample | 0.918 (dense dev) | **0.887** |
| dense (first try) | dense training, all context features | — | **0.83** |
| dense + fix | reverse-view context removed from dense training (§3.3) | — | **0.954** |
| + dev-tuned rule | decision rule tuned on the dense dev set | 0.9625 | **0.955** |
| v2 | website-name splitting, larger shortlists (top-K 20, min cheap 0.10), stage 3 | holdout B 0.9667; dev 0.9620 | first output not submitted; final **0.956125** |

---

# 2. Candidate Generation / Blocking Strategy

## 2.1 Stage A — multi-pass key blocking (`blocking.py`)
Each record emits a few keys built from its **rarest** cleaned name and address tokens (ranked by
IDF over the whole universe, so generic words such as "services", "limited" or "road" never form a
block). Every key is prefixed with the country label, so records only meet within a country. The
country is treated as an open set of strings — nothing is hard-coded to US/India, and France
passes through the same code.

| Key | Built from | Catches |
|---|---|---|
| `n1` | rarest name word | most matches with a distinctive name |
| `nn` | two rarest name words (sorted) | shuffled word order |
| `nc` | compact name (core words glued) | spacing differences (`Datech` / `Da Tech`) |
| `nf` | consonant skeleton of rarest name word + initial of the second | vowel typos |
| `nk` | order-free name key (sorted core words minus filler words) | `Akash Advisory Solutions` / `Solutions Advisory Akash Services` |
| `pn` | postcode + rarest name word | strong where postcodes exist (US ZIP, Indian PIN, French CP) |
| `a2` | two rarest address words | name rewritten / transliterated, address intact |
| `hs` | house number + rarest address word | same building, different name spelling |
| `na` | rarest name word + rarest address word | generic-ish names in a specific street |
| `ia` | name initials + rarest address word | abbreviated names (`HLE Sons` / `H L E`) |
| `np` | first 5 letters of compact name + rarest address word | truncated names |

Blocks larger than a size limit (300 target records / 20,000 pairs at training size, scaled with
the data size, capped at 4×) are dropped — they come from words that are not rare enough to be
informative.

## 2.2 Stage B — ranking and cut-off
The union of key hits is scored with a cheap character-n-gram TF-IDF cosine on name and address.
Per (Source 1 entity, source) we keep the top-K by a joint name+address score, plus the top-5 by
name cosine and the top-5 by address cosine, above a minimum cheap score.
Settings of the final run: top-K = 20 (15 in v1), minimum cheap score = 0.10 (0.15 in v1).

## 2.3 What is in `candidate_pairs.tsv`
Exactly the set the models run inference on — nothing is filtered afterwards. Stage 3 re-scores
the same pairs and does not add candidates, so every ID in `matching_results.tsv` is in
`candidate_pairs.tsv` (the official validator confirms this).

## 2.4 Blocking quality
| Measure | Value |
|---|---|
| Test Source 1 entities / Source 2+3 records | 1,732,544 / 9,969,589 |
| All possible pairs (all countries) | 1.727 × 10¹³ |
| Candidate pairs in `candidate_pairs.tsv` | 70,809,207 (≈40.9 per Source 1 entity) |
| Reduction ratio vs all pairs | 1 − 70.8M / 1.727×10¹³ = **0.999996** |
| Reduction ratio vs same-country pairs (6.72×10¹²) | **0.99999** |
| Blocking recall (true matches in the shortlist), dense training pairs of the final model | **0.9552** (41.0 candidates per entity) |

Blocking recall is the main ceiling of the system: with 95.5% of true matches in the shortlist,
even a perfect model would score about 0.99. See §4.6 and §4.8.

## 2.5 Scale
Prediction runs slice by slice and one country at a time (`predict_lowmem.py`). Because every key
contains the country, this gives exactly the same result as processing everything at once, and
runs on a 16 GB laptop.

---

# 3. Model Architecture and Feature Engineering

## 3.1 Cleaning (`normalization.py`) — the input to every feature
Fitted once on the training true pairs, saved in `norm_state.json` and inside `model.pkl`, never
refitted on test.

*Learned from the training data:* an Indian-script → Latin word dictionary (from true pairs where
the Source 2/3 name is in Devanagari, Tamil, Kannada, Malayalam, Telugu, …) with rule-based
transliteration as fallback; name and address abbreviation maps mined from true pairs; "protected"
real words that only look like typos (`pirate` vs `private`); the vocabulary of Source 1 name words.

*Names* → normalised name, **core** (legal forms removed wherever they appear), order-free **key**,
compact form, initials, set of legal forms. Handles DBA/aka aliases (`X dba Y` → `Y`), IDs/URLs/`M/s`
noise, accents, `&` vs `and`, dotted initials, digit-for-letter swaps (`Gu1f` → `gulf`), legal-word
typos (`Pirvate` → `private`), abbreviations (`Pvt`, `Ltd`, `Corp`, …), country words (`(India)`,
bare `France`), and **website names** (3.9% of Source 2/3 names, e.g. `urbanlakshmiservices.com`)
split into the most likely sequence of known Source 1 words → `urban laxmi services`.

*Addresses* → **postcode** (US ZIP, Indian PIN, French CP; extracted *before* the state, so
`CA 95113` leaves `CA`), **state/region code** (US states, Indian states in the codes Source 3
uses, French regions and every département), **house number + suffix** (`12 bis`, `7 ter`, `9 B`,
leading zeros removed), **street**, set of address components. `<NULL>`, `##`, `C/O`, `H.No`, PO boxes
and landmark words are handled; old/new city names are unified (`bombay` → `mumbai`).

*France (absent from training):* département → region table, French street types and their
misspellings right after the house number (`Avneue` → avenue, `COUR2` → cours, `Chm1` → chemin),
`l'`/`d'` with or without apostrophe (`l'Ecole` = `Lecole`), French legal forms (SARL, SAS, SASU,
EURL, SCI, …, `Compagnie`). All of these are generic rules; no French training labels exist.

## 3.2 Pair features (`features.py`, ≈76)
Structured comparisons are **1 when the fields agree, 0 when they explicitly conflict, NaN when
either side is missing**, so a missing postcode is never treated as a mismatch.

| Group | Features |
|---|---|
| Name string similarity | Jaro-Winkler, Levenshtein ratio, partial ratio, token-set and token-sort ratio, character TF-IDF cosine, exact core / key / normalised-name match, first/last token equal, initials equal, acronym match |
| Name structure | length ratio and difference, token-count difference, digit-count difference, alias flag, Indian-script flag, legal-form equal / conflict |
| Rarity | IDF of the rarest shared and rarest unshared name and address token (divided by log(N+1)), name frequency and address frequency in Source 1, "name is unique" |
| Address | token-set / token-sort ratio, cosine, exact match, component Jaccard, street similarity, house number / house digits / suffix / postcode / state agreement, number Jaccard and number conflicts, landmark flags, empty flags, length ratio |
| Cross checks | name-high-address-low, address-high-name-low, name-high-but-house-conflict, number of agreeing fields, number of conflicting fields |
| Blocking | which of the 11 key types fired, number of keys, cheap score and its rank, source (2 or 3) |

## 3.3 Models
- **Stage 1** — LightGBM (MIT) binary classifier on the pair features. Out-of-fold predictions are
  produced with folds grouped by Source 1 entity, so no entity leaks between folds.
- **Stage 2** — LightGBM on the pair features plus context computed from stage-1 scores within each
  (entity, source): rank, gap to the best, margin over the runner-up, number of candidates, sum of
  scores, number of high scores, and the best score from the other source.
  *Reverse-view context* ("how many other Source 1 entities picked this record, and is this one the
  best?") is **not** used in dense training: there only the sampled entities (~4.5% of Source 1)
  have shortlists, so a record almost never has a competitor, while in the test every entity
  competes. Using it dropped the leaderboard score to 0.83; removing it gave 0.954. Holdout scores
  could not reveal this, because holdouts had the same missing competition.
- **Calibration** — isotonic regression on holdout A (entity split 80% fit / 10% A / 10% B).
  Final model, dense holdouts: A 0.9694, **B 0.9667** (India 0.9658, US 0.9673); 4.12M training
  pairs from 100,000 sampled Source 1 entities.
- **Stage 3 (`stage3.py`)** — a small LightGBM trained on the **dense dev set**, where every entity
  is scored and records have their real competitors. Inputs: calibrated probability, its logit,
  and the full context *including* the reverse view, computed country by country (records only
  compete within a country). Dev entities are split by id hash 50% fit / 25% rule tuning / 25%
  check; stage 3 replaces the stage-2 output only if it scores higher on the check part.
  Result: "STAGE 3 GAIN on the check part: +0.0016", used: yes.

## 3.4 Decision rule (`decide.py`, `tune_on_dev.py`)
Two rule families: a global probability threshold, and an expected-F0.5 rule that picks, per
entity, the number of top candidates k (including k = 0) maximising expected F0.5. Either can be
combined with the **one-owner rule** (a Source 2/3 record goes to at most one Source 1 entity —
true for all 7.6M training pairs). 224 rules are searched on the dense dev set with vectorised
code; the winner is re-checked with the real decision code before it is saved.
Final rule: stage-3 expected-F0.5 rule `{'mode': 'expected_f', 'miss': 0.0, 'w0': 1.0, 'min_p': 0.4, 'single_owner': False}` on the stage-3 scores — check part F0.5 0.9633, precision 0.990, recall 0.917.

---

# 4. Other Relevant Information

## 4.1 Data analysis that shaped the design (training data)
| Finding | Number | Design consequence |
|---|---|---|
| Source 1 entities / true pairs | 2,206,821 / 7,638,365 (≈3.5 per entity) | several matches per source are normal → no per-source cap |
| S2/S3 records under two S1 entities | **0** of 7.6M | one-owner rule is always valid |
| `(ID: nnnnn)` in names | 0.3% of S2/S3, 56 true pairs with it on both sides | noise → removed, not a key |
| `#nnn` agreement on true pairs | 85–89%, mismatches mostly `012` vs `12` | leading zeros normalised |
| Website-style names | 3.9% of S2/S3, never in S1 | website names split into words |
| Same lower-cased name: true match? | 50.8% yes / 49.2% no | name alone is a coin flip → address decides |

## 4.2 Country as an open set / France
The country label is never a model feature and never filtered on. It is only used (a) as a prefix
of blocking keys, so records compare within the same country, and (b) to switch on country-specific
cleaning rules (e.g. French street-type typos). France, absent from training, runs through the same
code, and every test Source 1 entity — France included — gets exactly one output row.

## 4.3 Validation protocol
- All splits are **by Source 1 entity**; an entity and all its pairs are in exactly one split.
- Dense holdouts A (calibration, rule) and B (untouched score) inside dense training.
- Dense development set (test-like density, known answers) for rule tuning, stage 3 and error
  analysis. It reads slightly above the leaderboard (0.962 dev vs 0.954 leaderboard for the same
  model), mainly because France is not in the training data.
- Every output is checked by our own validator and the official `utils/validate_submission.py`
  (PASS required before upload).

## 4.4 Results
| Output | Dev F0.5 | Leaderboard |
|---|---|---|
| v1 tuned (`output_dense_tuned`) | 0.9625 | 0.955 |
| v2 first (`output_v2`) | 0.9615 | not submitted |
| v2 tuned (`output_v2_tuned`) | 0.9620 | not submitted |
| **v2 stage 3 (`output_v2_s3`, final)** | **0.9633** (check part; old score on the same part 0.9616) | **0.956125** |

## 4.5 Error analysis (`loss_breakdown.py`)
Every imperfect dev entity is put into exactly one bucket, and the points lost are added up (they
sum to 1 − dev F0.5): **A** singleton with a predicted match, **B** true matches never in the
shortlist, **C** matches in the shortlist but nothing predicted, **D** a wrong record predicted,
**E** all picks correct but some matches missed.

Tuned v2 rule on the full dev set; true matches never in the shortlist: 163,803 of 3,641,155.

| Bucket | ALL entities | ALL points | India entities | India points | US entities | US points |
|---|---|---|---|---|---|---|
| A singleton, predicted a match (false merge) | 2,666 | 0.0025 | 1,247 | 0.0030 | 1,419 | 0.0022 |
| B has matches, none in shortlist (blocking) | 3,953 | 0.0038 | 1,117 | 0.0027 | 2,836 | 0.0045 |
| C matches in shortlist, predicted none (too strict) | 3,951 | 0.0038 | 1,527 | 0.0036 | 2,424 | 0.0038 |
| D predicted a wrong record (false merge) | 21,058 | 0.0058 | 10,255 | 0.0070 | 10,803 | 0.0049 |
| E picks right, missed some (recall) | 262,891 | 0.0222 | 105,591 | 0.0216 | 157,300 | 0.0225 |
| **Dev F0.5** | 1,051,993 | **0.9620** | 421,114 | **0.9622** | 630,879 | **0.9619** |
| Blocking recall | | 0.9550 | | 0.9570 | | 0.9537 |

**Main finding:** precision ≈ 0.99, recall ≈ 0.92. Almost all remaining points are *missed*
matches, and most of those are lost in blocking (the true record never reaches the shortlist).

## 4.6 What we tried and rejected
- Using `(ID: nnn)` numbers or web domains as match keys — almost never present on both sides.
- An "at most one match per source" constraint — false in the data (≈3.5 matches per entity).
- Pretrained multilingual sentence embeddings (e.g. LaBSE, Apache-2.0 so allowed) — weak on
  character-level noise, prone to matching different businesses with similar generic names (costly
  under F0.5), and too large for our hardware; transliteration already covers cross-script names.
- Thin-sample training (0.887) and dense training with reverse-view context (0.83).

## 4.7 Engineering for limited hardware
All runs were done on 16 GB Windows laptops: prediction slice by slice per country, cleaning in
parallel worker processes with bounded caches, saved pair scores so a new decision rule can be
applied in minutes without re-scoring, and an unattended runner (`run_failsafe.py`) that resumes
after a crash without redoing finished steps.

## 4.8 Limitations and future work
1. **Blocking recall 0.955 is the ceiling.** Next steps: phonetic name keys, address-only keys
   (house number + street), postcode + house-number keys, MinHash/LSH on name character shingles,
   guided by `check_blocking.py` (which key types miss which matches).
2. A full-density dev set (all dev entities scored, not half) so stage 3 sees complete competition.
3. Averaging several dense models trained with different seeds / entity samples.
4. A small cross-encoder re-ranker for uncertain pairs only (needs a GPU).

## 4.9 Fair play and licences
No external databases, APIs, geocoding services or internet data are used; everything is learned
from the provided training data. The only built-in knowledge is generic: abbreviation and
legal-form lists and US / Indian state and French region/département names in `normalization.py`.
Models: LightGBM gradient-boosted trees (MIT licence), far below the 8-billion-parameter limit.
Other libraries: scikit-learn (BSD-3), RapidFuzz (MIT), pandas and numpy (BSD). No pretrained model
is used.

## 4.10 Reproducibility
Folder: `code/business_entity_resolution/` (`src/`, `README.md`, pinned `requirements.txt`).
```bash
pip install -r requirements.txt
# data in dataset/train and dataset/test, as in the student resource
python src/fix_source_files.py --train-dir dataset/train --test-dir dataset/test   # repair broken lines
python src/run_combined.py --n-jobs 4 --entities 100000 --tune --dev-frac 0.5 --stage3 --zip TEAM
python utils/validate_submission.py --matching output_dense_s3/matching_results.tsv \
       --candidate output_dense_s3/candidate_pairs.tsv --test-dir dataset/test
```
`run_combined.py` writes each stage's output to its own folder (`output_dense`, `…_tuned`, `…_s3`)
and builds the submission zip from the last accepted one. A full run takes about 6–8 hours on a
16 GB laptop. Every step and flag is described in `README.md`.
