"""
normalization.py - Step 2 of the business entity resolution pipeline.

v2 changes (search "v2" / read the notes below):
  * Replaces normalize.py + indic_latin.py; run_pipeline.py uses it directly (repr_frame()).
  * Addresses: postcode is cut FIRST, then the state -> "CA 95113", "Karnataka 560001",
    "DC 20500", "UP - 201301" keep their state; French "75001 Paris" is recognised.
  * Leading zeros kept on 5-digit postcodes (06000 Nice, 04101 Portland ME).
  * Split Indian PIN "Kolkata 700 001" -> 700001.  "Plot 5, ..." -> house 5.
  * Full French departement -> region table (France is only in TEST).  "Paris" stays a city.
  * Indian-script state words are turned into codes only in ADDRESSES, never in names
    ("बिहार स्टील" = "bihar steel", not "br steel").
  * Names from Indian scripts: "प्रा. लि." -> private limited, sound-alike legal words.
  * Typo fixer no longer eats real words ("Pirate Pizza"): words kept the same in true
    pairs are learned as protected.
  * name_legal is a plain string; repr frames keep business_name / business_address.

Turns one raw record (business_name, business_address, country) into the cleaned fields
that blocking and pair features compare. Standard library only; `unidecode` is used for
accent folding when installed (install it on BOTH the train and predict machine, or on neither).

Order of operations for every record:
  0. to_latin()      Indian-script words -> English letters (learned dictionary + letter rules),
                     Indian-script state names -> state codes.  Latin text passes through unchanged.
  1. pre-clean       name: "| www..", "X dba Y" -> Y, "(ID: 123)", "#12345", "M/s", "(India)".
                     address: <NULL>, N/A, "H.No", "N°", PO boxes, "#".
  2. base_clean()    accents, lowercase, & -> and, l.l.c. -> llc, no-436 -> no 436, hyphens.
  3. tokenize()      keeps 8-2-293/82/a as one token.
  4. _canon_token()  digit-for-letter repair (Gu1f -> gulf), ordinals -> digits,
                     learned abbreviation map, then the static tables below.
  5. name  -> legal suffixes found anywhere are split off, leading honorifics dropped.
     address -> split on commas into an unordered component set; state/region, postal code,
                house number and street pulled out.

Things learned from training data (Normalizer.fit) and saved with the model:
  - translit : Indian-script word -> English word   (LearnedDict)
  - name_map / addr_map : short -> long token maps (mine_abbreviations)
Nothing is ever refit on test data.  Country is an open set: no branch depends on its value.

CLI (see bottom of file):
  python normalization.py demo
  python normalization.py fit   --data-dir <dataset> --out work/norm_state.json [--sample-mod 8]
  python normalization.py apply --state work/norm_state.json --inp <file.tsv> --out <reps.tsv>
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict

try:  # better folding of accents / rare scripts; optional
    from unidecode import unidecode as _unidecode
except ImportError:  # pragma: no cover
    _unidecode = None

csv.field_size_limit(2**31 - 1)
STATE_VERSION = 4   # 4: website names split into words (vocab learned in fit); 3: 3: French fixes (glued l'/d', street typos, 7 ter, cite, bare 'France')

# =============================================================================
# Indian scripts -> English letters
# =============================================================================
# The nine Unicode Indic blocks (Devanagari 0900 ... Malayalam 0D00) share one layout, so a
# single table indexed by (codepoint & 0x7F) covers all of them.
INDIC_RE = re.compile(r"[ऀ-ൿ]")
_INDIC_WORD_RE = re.compile(r"[ऀ-ൿ‌‍]+")
_NORTH_BLOCKS = {0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00}  # final inherent 'a' is silent
_TAMIL, _GURMUKHI, _MALAYALAM, _BENGALI = 0x0B80, 0x0A00, 0x0D00, 0x0980

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL = {
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
    0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_SIGN = {
    0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4E: "e", 0x4F: "aw", 0x56: "ai", 0x57: "",
    0x62: "li", 0x63: "li",
}
_NUKTA_FIX = {"j": "z", "ph": "f", "d": "r", "dh": "rh", "k": "q", "g": "g", "kh": "kh"}
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}  # Malayalam
_LABIAL = {"p", "ph", "b", "bh", "m"}

# State names as they appear in the S2/S3 addresses (all 16 seen in training), mapped to the
# same 2-letter codes Source 3 uses.  Matching is done on NFC-normalised text.
INDIC_STATE = {
    "महाराष्ट्र": "MH", "दिल्ली": "DL", "उत्तर प्रदेश": "UP", "ಕರ್ನಾಟಕ": "KA",
    "தமிழ்நாடு": "TN", "পশ্চিমবঙ্গ": "WB", "ગુજરાત": "GJ", "తెలంగాణ": "TG",
    "हरियाणा": "HR", "राजस्थान": "RJ", "കേരളം": "KL", "बिहार": "BR",
    "मध्य प्रदेश": "MP", "ఆంధ్రప్రదేశ్": "AP", "ਪੰਜਾਬ": "PB", "ଓଡ଼ିଶା": "OD",
}
_INDIC_STATE_NFC = sorted(
    ((unicodedata.normalize("NFC", k), v) for k, v in INDIC_STATE.items()),
    key=lambda kv: -len(kv[0]),
)


def _indic_word_translit(word: str) -> str:
    """Letter-by-letter transliteration of one Indic word (no dictionary)."""
    out: list[str] = []
    pending = False  # last output is a consonant still waiting for its vowel
    block = 0x0900
    n = len(word)
    i = 0
    while i < n:
        cp = ord(word[i])
        if not (0x0900 <= cp <= 0x0D7F):
            if pending:
                out.append("a")
                pending = False
            if cp not in (0x200C, 0x200D):
                out.append(word[i])
            i += 1
            continue
        block, off = cp & ~0x7F, cp & 0x7F
        nxt = ord(word[i + 1]) & 0x7F if i + 1 < n and 0x0900 <= ord(word[i + 1]) <= 0x0D7F else -1
        if block == _MALAYALAM and off in _CHILLU:
            if pending:
                out.append("a")
            out.append(_CHILLU[off])
            pending = False
        elif block == _BENGALI and off == 0x4E:  # khanda ta
            if pending:
                out.append("a")
            out.append("t")
            pending = False
        elif block == _GURMUKHI and off == 0x70:  # tippi
            if pending:
                out.append("a")
            out.append("n")
            pending = False
        elif block == _GURMUKHI and off == 0x71:  # addak (gemination)
            if pending:
                out.append("a")
            pending = False
        elif block == _TAMIL and off == 0x03 and nxt == 0x2A:  # aytham + pa = f
            if pending:
                out.append("a")
            out.append("f")
            pending = True
            i += 2
            continue
        elif off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off == 0x3C:  # nukta modifies the previous consonant
            if out and out[-1] in _NUKTA_FIX:
                out[-1] = _NUKTA_FIX[out[-1]]
        elif off == 0x4D:  # virama: no vowel
            pending = False
        elif off in _SIGN:
            out.append(_SIGN[off])
            pending = False
        elif off in _VOWEL:
            if pending:
                out.append("a")
            out.append(_VOWEL[off])
            pending = False
        elif off in (0x01, 0x02):  # candrabindu, anusvara
            if pending:
                out.append("a")
            nx = _CONS.get(nxt, "")
            out.append("m" if nx in _LABIAL else "n")
            pending = False
        elif off == 0x03:  # visarga
            if pending:
                out.append("a")
            out.append("h")
            pending = False
        elif 0x66 <= off <= 0x6F:  # digits
            if pending:
                out.append("a")
            out.append(str(off - 0x66))
            pending = False
        elif off == 0x50:
            out.append("om")
            pending = False
        # everything else (danda, avagraha, length marks ...) is dropped
        i += 1
    if pending and block not in _NORTH_BLOCKS:
        out.append("a")
    return "".join(out)


_TRANSLIT_CACHE: dict[str, str] = {}

# Max entries per cache, PER PROCESS.  Each worker has its own copies, so with 8 workers the
# defaults can cost several GB.  set_cache_limits("low") for small machines (a little slower).
CACHE_LIMITS = {"translit": 500_000, "name": 1_000_000, "addr": 2_000_000, "comp": 2_000_000}
_LOW_LIMITS = {"translit": 100_000, "name": 150_000, "addr": 250_000, "comp": 250_000}


def set_cache_limits(mode: str = "low"):
    CACHE_LIMITS.update(_LOW_LIMITS if mode == "low" else
                        {"translit": 500_000, "name": 1_000_000, "addr": 2_000_000, "comp": 2_000_000})


def rule_translit(word: str) -> str:
    r = _TRANSLIT_CACHE.get(word)
    if r is None:
        r = _indic_word_translit(word)
        if len(_TRANSLIT_CACHE) < CACHE_LIMITS["translit"]:
            _TRANSLIT_CACHE[word] = r
    return r


_PK_SUBS = [("chh", "C"), ("ch", "C"), ("sh", "s"), ("ph", "f"), ("bh", "b"), ("kh", "k"),
            ("gh", "g"), ("th", "t"), ("dh", "d"), ("jh", "j"), ("ck", "k"), ("q", "k"),
            ("x", "ks"), ("w", "b"), ("v", "b"), ("z", "j"), ("c", "k"), ("C", "c")]


def phonetic_key(s: str) -> str:
    """Sound-alike key: first letter + consonant skeleton (private / praivet / praibhet -> prbt)."""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    for a, b in _PK_SUBS:
        s = s.replace(a, b)
    body = [ch for ch in s[1:] if ch not in "aeiouy"]
    out = [s[0]]
    for ch in body:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _key_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 - _lev(a, b) / max(len(a), len(b))


_EN_WORD_RE = re.compile(r"[a-z0-9]+")
_ANY_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ɏऀ-ൿ‌‍]+")


def _prep_indic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.replace("‌", "").replace("‍", "")


class LearnedDict:
    """Indic word -> English word, learned from true-match pairs (S1 English name, S2/S3 Indic name)."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self.map: dict[str, str] = dict(mapping or {})

    def fit(self, pairs, min_count: int = 3, min_share: float = 0.5, min_sim: float = 0.45):
        votes: dict[str, Counter] = defaultdict(Counter)
        for eng, ind in pairs:
            E = _EN_WORD_RE.findall(ascii_fold(eng).lower())
            W = _ANY_WORD_RE.findall(_prep_indic(ind))
            if not E or not W:
                continue
            if len(E) == len(W):  # same word count: align by position
                for e, w in zip(E, W):
                    if INDIC_RE.search(w):
                        votes[w][e] += 1
                continue
            ekeys = [(e, phonetic_key(e)) for e in E]
            for w in W:  # otherwise: closest-sounding English word
                if not INDIC_RE.search(w):
                    continue
                wk = phonetic_key(rule_translit(w))
                best, bs = None, min_sim
                for e, ek in ekeys:
                    s = _key_sim(wk, ek)
                    if s > bs:
                        best, bs = e, s
                if best:
                    votes[w][best] += 1
        self.map = {}
        for w, c in votes.items():
            e, k = c.most_common(1)[0]
            tot = sum(c.values())
            if k >= min_count and k / tot >= min_share:
                self.map[w] = e
        return self

    def to_latin(self, text: str, states: bool = True) -> tuple[str, int, int]:
        """Returns (text in English letters, dictionary hits, rule hits). Non-Indic text is untouched.
        states=True turns Indian-script state names into codes: addresses only. In a NAME
        ("बिहार स्टील" = Bihar Steel) the state word is part of the business name and must stay a word."""
        if not text or not INDIC_RE.search(text):
            return text, 0, 0
        text = _prep_indic(text)
        if states:
            for k, code in _INDIC_STATE_NFC:
                if k in text:
                    text = text.replace(k, f" {code} ")
        hits = [0, 0]

        def sub(m):
            w = m.group()
            e = self.map.get(w)
            if e is not None:
                hits[0] += 1
                return e
            hits[1] += 1
            return rule_translit(w)

        text = _INDIC_WORD_RE.sub(sub, text)
        return re.sub(r"\s+", " ", text).strip(), hits[0], hits[1]

    def state(self) -> dict:
        return {"map": self.map}

    @classmethod
    def from_state(cls, st: dict | None) -> "LearnedDict":
        return cls((st or {}).get("map"))


# =============================================================================
# Static tables
# =============================================================================
COUNTRY_ALIASES = {
    "us": {"us", "usa", "u s", "u s a", "united states", "united states of america", "america"},
    "in": {"in", "india", "ind", "bharat", "republic of india"},
    "fr": {"fr", "fra", "france", "republique francaise"},
}
_COUNTRY_LOOKUP = {a: code for code, al in COUNTRY_ALIASES.items() for a in al}

# Legal forms (after canonicalisation). Removed from name_core wherever they appear,
# because S2/S3 shuffle word order ("SOLUTIONS LIMITED ADVISORY PRIVATE AKASH").
LEGAL = {
    # US / UK / generic
    "llc", "incorporated", "corporation", "company", "limited", "lp", "llp", "lllp", "pc",
    "pllc", "plc", "pa", "lc", "ltda", "gmbh", "ag", "bv", "nv", "spa", "srl", "pty",
    # India
    "private", "opc",
    # France (and other civil-law forms)
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scs", "sca", "scop", "scp", "sel",
    "selarl", "selas", "ei", "eirl", "earl", "gie", "sem",
}
NAME_STOP = {"the", "of", "and", "a", "an", "de", "des", "du", "la", "le", "les", "d", "l", "for"}
HONORIFIC = {"mr", "mrs", "dr", "smt", "shri", "messrs", "kumari", "sh"}  # leading only

ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}

NAME_ABBREV = {
    "inc": "incorporated", "incorp": "incorporated", "corp": "corporation", "co": "company",
    "cie": "company", "ltd": "limited", "ltda": "limited", "lmtd": "limited", "ltd.": "limited",
    "pvt": "private", "prvt": "private", "pte": "private", "bros": "brothers", "bro": "brothers",
    "intl": "international", "natl": "national", "mfg": "manufacturing", "assoc": "associates",
    "assocs": "associates", "assn": "association", "svc": "services", "svcs": "services",
    "service": "services", "mgmt": "management", "dept": "department", "ctr": "center",
    "cntr": "center", "centre": "center", "univ": "university", "hosp": "hospital",
    "engg": "engineering", "ents": "enterprises", "inds": "industries", "ets": "etablissements",
    "groupe": "group", "compagnie": "company", "et": "and", "n": "and",
    "st": "st", "saint": "st", "sainte": "ste", "mt": "mount", "ft": "fort",
    "doctor": "dr", "docteur": "dr",
    # Shree / Sree / Sri / Shri are spelling variants; a leading one is noise (see HONORIFIC)
    "shree": "shri", "sree": "shri", "sri": "shri", "shri": "shri",
    # Indian-name spelling variants left over after transliteration
    "lakshmi": "laxmi", "luxmi": "laxmi", "jay": "jai",
    **ORDINAL_WORDS,
}

ADDR_ABBREV = {
    # street / saint / suite are ambiguous across countries -> merged tokens
    "st": "st", "street": "st", "str": "st", "strt": "st", "saint": "st",
    "sainte": "ste", "ste": "ste", "suite": "ste",
    "dr": "dr", "drive": "dr", "drv": "dr", "doctor": "dr", "docteur": "dr",
    "pl": "pl", "place": "pl", "plot": "pl",
    "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue", "aven": "avenue",
    "ln": "lane", "ct": "court", "crt": "court", "cir": "circle", "circ": "circle",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "boul": "boulevard", "blv": "boulevard",
    "trl": "trail", "twp": "township", "hwy": "highway", "hiway": "highway", "ter": "terrace",
    "terr": "terrace", "pkwy": "parkway", "pky": "parkway", "ft": "fort", "mt": "mount",
    "mtn": "mountain", "hts": "heights", "sq": "square", "sqr": "square", "cty": "county",
    "apt": "apartment", "apts": "apartment", "appt": "apartment", "apartments": "apartment",
    "appartement": "apartment", "bldg": "building", "bld": "building", "bdg": "building",
    "batiment": "building", "bat": "building", "fl": "floor", "flr": "floor", "etage": "floor",
    "etg": "floor", "flt": "flat", "rm": "room", "ofc": "office",
    "nr": "near", "opp": "opposite", "sec": "sector", "sect": "sector", "extn": "extension",
    "ext": "extension", "ngr": "nagar", "soc": "society", "hsg": "housing",
    # France
    "r": "rue", "all": "allee", "imp": "impasse", "rte": "route", "ch": "chemin",
    "chem": "chemin", "crs": "cours", "q": "quai", "res": "residence", "resid": "residence",
    "psg": "passage", "lot": "lotissement", "lotiss": "lotissement", "chs": "chaussee",
    "fbg": "faubourg", "prom": "promenade", "rpt": "rondpoint",
    "cour": "cours", "cour2": "cours", "chm": "chemin", "chm1": "chemin", "alle": "allee",
    # old / new city names (seen as swaps between S1 and S2/S3 in training)
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "poona": "pune",
    "bengaluru": "bangalore", "gurugram": "gurgaon",
    **ORDINAL_WORDS,
}
ADDR_DROP = {"null", "na", "none", "cdp", "divreportingcircle", "region", "hq", "cedex",
             "no", "number", "num", "hno", "hn", "door", "city", "www"}
ADDR_STOP = {"de", "des", "du", "la", "le", "les", "d", "l", "of", "the", "and", "et"}
STREET_TYPES = {
    "st", "road", "dr", "avenue", "lane", "court", "circle", "pl", "boulevard", "trail",
    "highway", "terrace", "parkway", "way", "rue", "allee", "impasse", "route", "chemin",
    "cours", "quai", "square", "passage", "lotissement", "chaussee", "marg", "cross", "path",
    "alley", "row", "loop", "pike", "plaza", "faubourg", "promenade", "rondpoint", "mail",
    "esplanade", "sentier", "salai", "veedhi", "gali",
    # French places that take a house number like a street ("9 B Cite Mouneyra")
    "cite", "residence", "hameau", "ruelle", "voie", "sente", "montee", "traverse", "rampe", "parvis",
}
# Common French words written with l'/d' in front.  Sources write "l'Ecole", "L Ecole" AND
# "Lecole": the apostrophe form is cut to "ecole" by base_clean; this list un-glues "lecole" too.
# Only words of 4+ letters.  Left out on purpose: place names and surnames that start with L/D
# ("Lorient", "Lange", "Leclair", "Darts", "Daube"), which would be cut wrongly.
_FR_ELIDED = {
    "ecole", "ecoles", "entente", "eglise", "europe", "etoile", "etoiles", "esperance", "union",
    "avenir", "atelier", "ateliers", "amicale", "hotel", "habitat", "harmonie", "ocean", "olivier",
    "eleve", "eleves", "industrie", "espace", "abbaye", "etang", "image", "immobilier", "univers",
    "enfance", "emploi", "employeurs", "epicerie", "enfant", "enfants", "education", "equipe",
    "erable", "ermitage", "hermitage", "hopital", "horloge", "innovation", "institut", "ecluse",
    "eden", "energie", "environnement", "entreprise", "entreprises", "artisan", "artisanat",
    "atlantique", "amitie", "autonomie", "automobile", "orangerie", "oasis", "opera", "eglantine",
    "yser", "eveil", "esprit", "etudes", "ecoute", "economie", "habitation", "hirondelle",
    "humanite", "abri", "accueil", "agriculture", "alliance", "ancre", "arsenal", "assurance",
    "audace", "estuaire", "armee", "arbre", "aqueduc", "ouest",
}
_FR_ELIDED = {w for w in _FR_ELIDED if len(w) >= 4}
UNIT_WORDS = {"floor", "apartment", "ste", "unit", "flat", "room", "building",
              "office", "shop", "pobox", "tower", "wing", "level", "gf", "ff", "sf"}
HOUSE_SFX = {"bis", "ter", "quater"}
# Filler words S2/S3 append to names ("Sapphire LLC Services", "Delhi Limited Center").
# name_core keeps them; name_key drops them so it can be used as an exact-match / blocking key.
NAME_FILLER = {"services", "center", "partners", "group", "holdings", "enterprises", "labs",
               "sys", "district", "board", "council", "authority", "commission", "federation"}

# ---- states / regions: whole address components only --------------------------------------
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi", "american samoa": "as",
}
_IN_STATES = {  # codes as written by Source 3
    "maharashtra": "mh", "delhi": "dl", "nct of delhi": "dl", "uttar pradesh": "up",
    "karnataka": "ka", "tamil nadu": "tn", "tamilnadu": "tn", "gujarat": "gj",
    "west bengal": "wb", "telangana": "tg", "haryana": "hr", "rajasthan": "rj", "kerala": "kl",
    "keralam": "kl", "bihar": "br", "madhya pradesh": "mp", "andhra pradesh": "ap",
    "punjab": "pb", "orissa": "od", "odisha": "od", "assam": "as", "goa": "ga",
    "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "uttaranchal": "uk",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "tripura": "tr",
    "meghalaya": "ml", "manipur": "mn", "nagaland": "nl", "mizoram": "mz",
    "arunachal pradesh": "ar", "sikkim": "sk",
}
_FR_REGION_MEMBERS = {  # region code -> region name, pre-2016 region names, and every departement
    "hdf": ["hauts de france", "nord pas de calais", "picardie", "nord", "pas de calais", "somme",
            "oise", "aisne"],
    "naq": ["nouvelle aquitaine", "aquitaine", "limousin", "poitou charentes", "gironde", "landes",
            "pyrenees atlantiques", "lot et garonne", "dordogne", "charente", "charente maritime",
            "deux sevres", "vienne", "haute vienne", "creuse", "correze"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["ile de france", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise", "val doise"],
    "ara": ["auvergne rhone alpes", "auvergne", "rhone alpes", "ain", "allier", "ardeche", "cantal",
            "drome", "isere", "loire", "haute loire", "puy de dome", "rhone", "savoie", "haute savoie",
            "metropole de lyon"],
    "bfc": ["bourgogne franche comte", "bourgogne", "franche comte", "cote d or", "cote dor", "doubs",
            "jura", "nievre", "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "cotes darmor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "centre", "cher", "eure et loir", "indre", "indre et loire",
            "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "alsace", "lorraine", "champagne ardenne", "ardennes", "aube", "marne",
            "haute marne", "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "nor": ["normandie", "basse normandie", "haute normandie", "calvados", "eure", "manche", "orne",
            "seine maritime"],
    "occ": ["occitanie", "languedoc roussillon", "midi pyrenees", "ariege", "aude", "aveyron", "gard",
            "haute garonne", "gers", "herault", "lot", "lozere", "hautes pyrenees",
            "pyrenees orientales", "tarn", "tarn et garonne"],
    "pac": ["provence alpes cote azur", "provence alpes cote d azur", "paca",
            "alpes de haute provence", "hautes alpes", "alpes maritimes", "bouches du rhone", "var",
            "vaucluse"],
}
_FR_REGIONS = {name: code for code, names in _FR_REGION_MEMBERS.items() for name in names}
ADMIN: dict[str, str] = {}
for _tbl in (_US_STATES, _IN_STATES, _FR_REGIONS):
    ADMIN.update(_tbl)
STATE_CODES = set(_US_STATES.values()) | set(_IN_STATES.values())
ADMIN.update({c: c for c in STATE_CODES})
ADMIN.update({c: c for c in set(_FR_REGIONS.values())})


# =============================================================================
# Cleaning primitives
# =============================================================================
_PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "´": "'", "`": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ", "﻿": " ",
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ø": "o", "Ø": "o", "đ": "d", "ł": "l", "Ł": "l",
})


def ascii_fold(s: str) -> str:
    """Unicode -> ASCII (é -> e). Run to_latin() first: Indic text must never reach this."""
    if s.isascii():
        return s
    s = s.translate(_PUNCT_MAP)
    if _unidecode is not None:
        return _unidecode(s)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


_DOTTED_RE = re.compile(r"\b(?:[a-z]\.\s?){2,}(?:[a-z]\b)?|\b[a-z]\.[a-z]\b")
_ELISION_RE = re.compile(r"\b[dl]'(?=[a-z])")
_APOS_RE = re.compile(r"'")
_WORD_DIGIT_RE = re.compile(r"(?<=[a-z]{2})-(?=\d)")
_LETTER_DIGIT_RE = re.compile(r"\b([a-z])-(?=\d)")
_LETTER_HYPHEN_RE = re.compile(r"(?<=[a-z])-(?=[a-z])|(?<=\d)-(?=[a-z]{2})")
_JUNK_RE = re.compile(r"[^a-z0-9/\- ]+")
_WS_RE = re.compile(r"\s+")


def base_clean(s: str) -> str:
    """accents, lowercase, & -> and, dotted initials joined, letter-dash-digit split."""
    if not s:
        return ""
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ").replace("+", " and ")
    s = _DOTTED_RE.sub(lambda m: m.group().replace(".", "").replace(" ", "") + " ", s)
    s = _ELISION_RE.sub("", s)          # l'europe -> europe
    s = _APOS_RE.sub("", s)             # gabriela's -> gabrielas
    s = _WORD_DIGIT_RE.sub(" ", s)      # no-436 -> no 436
    s = _LETTER_DIGIT_RE.sub(r"\1", s)  # b-195 -> b195
    s = _LETTER_HYPHEN_RE.sub(" ", s)   # baule-escoublac -> baule escoublac
    s = _JUNK_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokenize(s: str) -> list[str]:
    """Split on spaces; '/' and '-' survive only inside tokens with a digit (8-2-293/82/a)."""
    out = []
    for t in s.split():
        t = t.strip("/-")
        if not t:
            continue
        if ("/" in t or "-" in t) and not any(ch.isdigit() for ch in t):
            out.extend(x for x in re.split(r"[/-]+", t) if x)
        else:
            out.append(t)
    return out


_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th|er|re|eme|e|ere)$")
_LEET = {"0": "o", "1": "l", "5": "s", "6": "g", "8": "b"}
_LEET_MID_RE = re.compile(r"(?<=[a-z])[01568](?=[a-z])")
_LEET_HEAD_RE = re.compile(r"^[01568](?=[a-z]{2})")
_LEET_TAIL_RE = re.compile(r"(?<=[a-z]{2})[01568]$")


def _repair_digits(t: str, tail: bool) -> str:
    """Gu1f -> gulf, 8rothers -> brothers, denta1 -> dental (names only for the tail case)."""
    if t.isalpha() or t.isdigit() or _ORDINAL_RE.match(t):
        return t
    if re.search(r"\d\d", t) or not re.fullmatch(r"[a-z01568]+", t):
        return t
    t2 = _LEET_MID_RE.sub(lambda m: _LEET[m.group()], t)
    t2 = _LEET_HEAD_RE.sub(lambda m: _LEET[m.group()], t2)
    if tail:
        t2 = _LEET_TAIL_RE.sub(lambda m: _LEET[m.group()], t2)
    return t2


# words whose typos matter most (they decide what name_core keeps)
_TYPO_TARGETS = ["private", "limited", "incorporated", "corporation", "company", "associates",
                 "services", "brothers", "industries", "enterprises", "solutions",
                 "technologies", "international", "foundation"]


def _osa(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance with early exit above cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        best = cap + 1
        for j in range(1, len(b) + 1):
            c = a[i - 1] != b[j - 1]
            v = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, d[i - 2][j - 2] + 1)
            d[i][j] = v
            best = min(best, v)
        if best > cap:
            return cap + 1
    return d[-1][-1]


@functools.lru_cache(maxsize=1_000_000)
def _fix_typo(t: str) -> str:
    if len(t) < 5 or not t.isalpha():
        return t
    for w in _TYPO_TARGETS:
        cap = 2 if len(w) >= 9 or w == "private" else 1   # pirvte / prite -> private
        if t[0] == w[0] and _osa(t, w, cap) <= cap:
            return w
    return t


def _split_doubled(t: str) -> str:
    """llcllc -> llc, pcpc -> pc."""
    h = len(t) // 2
    if len(t) % 2 == 0 and h >= 2 and t[:h] == t[h:] and t[:h] in LEGAL | set(NAME_ABBREV):
        return t[:h]
    return t


def _unglue(t: str) -> str:
    """lecole -> ecole, deleves -> eleves, lyser -> yser (only words in _FR_ELIDED)."""
    if len(t) >= 5 and t[0] in "ld" and t[1:] in _FR_ELIDED:
        return t[1:]
    return t


# French street types and how they get misspelt.  Used ONLY for French addresses and ONLY on the
# word right after the house number ("41 Avneue ...", "5 Bdulevard ...", "12 Coir ..."), because
# that is where a street type must be; elsewhere "Allen" or "Coors" are real names.
_STREET_TYPO_TARGETS = {"avenue": "avenue", "boulevard": "boulevard", "allee": "allee",
                        "impasse": "impasse", "chemin": "chemin", "route": "route",
                        "passage": "passage", "cours": "cours", "cour": "cours",
                        "residence": "residence", "lotissement": "lotissement",
                        "promenade": "promenade", "chaussee": "chaussee", "faubourg": "faubourg"}


@functools.lru_cache(maxsize=200_000)
def _fix_street(t: str) -> str:
    if len(t) < 4 or not t.isalpha() or t in STREET_TYPES:
        return t
    for w, canon in _STREET_TYPO_TARGETS.items():
        cap = 2 if len(w) >= 8 and len(t) >= 7 else 1
        if t[0] == w[0] and _osa(t, w, cap) <= cap:
            return canon
    return t


_HOUSE_NUM_RE = re.compile(r"^[a-z]?\d[0-9a-z/-]*$")
_BARE_COUNTRY_WORDS = {"fr": {"france"}}   # "Marina Ecole France Sarl" vs "Marina Ecole (France) Sarl"


def normalize_country(c: str) -> str:
    """Open set: known aliases collapse (usa -> us), anything else is kept as a folded slug."""
    k = base_clean(c or "").replace("-", " ")
    k = _WS_RE.sub(" ", k).strip()
    return _COUNTRY_LOOKUP.get(k, k.replace(" ", "_"))


def _country_words(country_norm: str) -> set[str]:
    words = {country_norm.replace("_", " ")}
    for code, al in COUNTRY_ALIASES.items():
        if code == country_norm:
            words |= al
    return words


# =============================================================================
# Abbreviation mining (training only)
# =============================================================================
def _is_abbrev(short: str, long: str) -> bool:
    if len(short) < 2 or len(short) >= len(long) or short[0] != long[0]:
        return False
    if not short.isalpha() or not long.isalpha():
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def mine_abbreviations(left, right, min_count: int = 25, protected: set | None = None,
                       min_share: float = 0.5) -> dict[str, str]:
    """left/right: parallel lists of token lists from true-match pairs -> {short: long}."""
    protected = protected or set()
    cnt: Counter = Counter()
    seen: Counter = Counter()  # pairs containing the token at all
    for L, R in zip(left, right):
        A, B = set(L), set(R)
        seen.update(A | B)
        oa, ob = A - B, B - A
        if not oa or not ob:
            continue
        for xs, ys in ((oa, ob), (ob, oa)):
            for s in xs:
                if s in protected:
                    continue
                for lg in ys:
                    if _is_abbrev(s, lg):
                        cnt[s, lg] += 1
    per_short: dict[str, Counter] = defaultdict(Counter)
    for (s, lg), n in cnt.items():
        per_short[s][lg] = n
    out = {}
    for s, c in per_short.items():
        lg, n = c.most_common(1)[0]
        # the short form must mostly *be* an abbreviation: rejects york -> yorktown, ear -> earnosethroat
        if (n >= min_count and n / sum(c.values()) >= min_share and lg not in per_short
                and n / seen[s] >= 0.2):
            out[s] = lg
    return out


# =============================================================================
# The normaliser
# =============================================================================
_PIPE_RE = re.compile(r"\s*\|.*$")
_ALIAS_RE = re.compile(  # "X dba Y": Y is the name S1 carries (6,686 of 6,686 training cases)
    r"(?:\bd\s?[./]?\s?b\s?[./]?\s?a\b\.?|\bdoing\s+business\s+as\b|\bformerly\s+known\s+as\b"
    r"|\bformerly\b|\bf\s?[./]?\s?k\s?[./]?\s?a\b\.?|\ba\s?[./]\s?k\s?[./]\s?a\b\.?|\baka\b"
    r"|\balso\s+known\s+as\b|\btrading\s+as\b|\bt\s?/\s?a\b)\s*:?\s*",
    re.I,
)
_ID_RE = re.compile(r"[(\[]\s*id\s*[:#.]?\s*\d+\s*[)\]]|\bid\s*[:#]\s*\d+|#\s*\d{3,}", re.I)
_MS_RE = re.compile(r"^\W*(?:m\s*/\s*s|messrs)\b\.?", re.I)
_URL_HEAD_RE = re.compile(r"\b(?:https?://)?www\.", re.I)
_TLD_RE = re.compile(r"(?<=[a-z0-9])\.(?:co\.in|c0m|com|net|org|in|fr|biz|info|us)\b", re.I)
_PAREN_RE = re.compile(r"[(\[{]\s*([^()\[\]{}]*?)\s*[)\]}]")
_LEAD_RE = re.compile(r"^[^0-9A-Za-zÀ-ɏ]+")

_A_NULL_RE = re.compile(r"<\s*null\s*>|\bn\s*/\s*a\b", re.I)
_A_POBOX_RE = re.compile(r"\bp\.?\s?o\.?\s*box\b|\bpmb\b|\bb\.?\s?p\.?\s*(?=\d)|\bpost\s+box\b", re.I)
_A_HNO_RE = re.compile(r"\bh\s*\.?\s*no\b\.?|\bd\s*\.\s*no\b\.?|\bn\s*[°º]\s*|№", re.I)
_A_CARE_RE = re.compile(r"\b[csdw]\s*/\s*o\b", re.I)  # c/o, s/o, d/o, w/o
_A_LAYOUT_RE = re.compile(r"\bl\s*/\s*o\b", re.I)


class Normalizer:
    """Holds the learned maps; name_repr / addr_repr / record() are pure functions of the input."""

    def __init__(self, translit: LearnedDict | None = None, name_map=None, addr_map=None, protect=None,
                 vocab=None):
        self.translit = translit or LearnedDict()
        # word -> count over Source 1 names; used to split website names ("urbanlakshmiservices")
        self.vocab: dict[str, int] = dict(vocab or {})
        self._vlog_n = math.log(sum(self.vocab.values()) + 1) if self.vocab else 0.0
        self._seg_cache: dict[str, tuple] = {}
        self.name_map: dict[str, str] = dict(name_map or {})
        self.addr_map: dict[str, str] = dict(addr_map or {})
        # real words that look like typos of a legal word (pirate ~ private): never "fixed"
        self.protect: set[str] = set(protect or ()) | _TYPO_PROTECT_SEED
        self._name_tok_cache: dict[str, str] = {}
        self._addr_tok_cache: dict[str, str] = {}
        self._comp_cache: dict[str, tuple] = {}

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"version": STATE_VERSION, "translit": self.translit.state(),
                "name_map": self.name_map, "addr_map": self.addr_map,
                "protect": sorted(self.protect - _TYPO_PROTECT_SEED),
                "vocab": self.vocab,
                "unidecode": _unidecode is not None}

    @classmethod
    def from_state(cls, st: dict | None) -> "Normalizer":
        st = st or {}
        if st.get("unidecode") is not None and st["unidecode"] != (_unidecode is not None):
            print("WARNING: unidecode availability differs from training; "
                  "accent folding may not be identical.", file=sys.stderr)
        return cls(LearnedDict.from_state(st.get("translit")), st.get("name_map"), st.get("addr_map"),
                   st.get("protect"), st.get("vocab"))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_state(json.load(f))

    # ---------------------------------------------------------------- tokens
    def _canon_name_token(self, t: str) -> str:
        r = self._name_tok_cache.get(t)
        if r is not None:
            return r
        r = _split_doubled(_repair_digits(t, tail=True))
        if r[:2] in ("ln", "lm") and len(r) >= 3 and r.isalpha():  # lnc -> inc, lmpex -> impex
            r = "i" + r[1:]
        m = _ORDINAL_RE.match(r)
        if m:
            r = m.group(1)
        r = _unglue(r)
        r = self.name_map.get(r, r)
        r = NAME_ABBREV.get(r, r)
        if r not in LEGAL and r not in NAME_ABBREV.values() and r not in self.protect:
            f = _fix_typo(r)
            r = NAME_ABBREV.get(f, f)
        if r.isdigit():
            r = r.lstrip("0") or "0"
        if len(self._name_tok_cache) < CACHE_LIMITS["name"]:
            self._name_tok_cache[t] = r
        return r

    def _canon_addr_token(self, t: str) -> str:
        r = self._addr_tok_cache.get(t)
        if r is not None:
            return r
        r = _repair_digits(t, tail=False) if len(t) >= 5 else t
        m = _ORDINAL_RE.match(r)
        if m:
            r = m.group(1)
        r = _unglue(r)
        r = self.addr_map.get(r, r)
        r = ADDR_ABBREV.get(r, r)
        if r.isdigit() and not _KEEP_ZERO_RE.match(r):
            r = r.lstrip("0") or "0"   # 00109 -> 109, but 06000 / 04101 are postcodes: keep
        if len(self._addr_tok_cache) < CACHE_LIMITS["addr"]:
            self._addr_tok_cache[t] = r
        return r

    # ---------------------------------------------------------------- names
    def name_repr(self, raw: str, country: str = "") -> dict:
        raw = raw or ""
        was_indic = bool(INDIC_RE.search(raw))
        s, _, _ = self.translit.to_latin(raw, states=False)
        s = s.translate(_PUNCT_MAP)
        s2 = _PIPE_RE.sub("", s)                      # "Name | www.name.com"
        s = s2 if s2.strip() else s.replace("|", " ")
        had_alias = 0
        ms = list(_ALIAS_RE.finditer(s))              # "Solmira DBA Eastern Mill" -> "Eastern Mill"
        for m in reversed(ms):
            if s[: m.start()].strip() and s[m.end():].strip():
                s = s[m.end():]
                had_alias = 1
                break
        s = _ID_RE.sub(" ", s)                        # (ID: 37946), #7525
        s = _MS_RE.sub(" ", s)                        # M/s
        was_site = bool(_URL_HEAD_RE.search(s) or _TLD_RE.search(s))
        s = _URL_HEAD_RE.sub("", s)
        s = _TLD_RE.sub("", s)                        # familystarhealth.com -> familystarhealth
        cw = _country_words(normalize_country(country)) if country else set()

        def paren(m):                                 # drop "(India)", "(Frànce)"; keep "(LLC)" text
            inner = base_clean(m.group(1))
            if inner.startswith("ln"):
                inner = "in" + inner[2:]
            return " " if inner in cw else f" {m.group(1)} "

        s = _PAREN_RE.sub(paren, s)
        s = _LEAD_RE.sub("", s.replace("#", " ").replace("@", " "))
        c = base_clean(s)
        toks = tokenize(c)
        if len(toks) == 1 and len(toks[0]) >= 10 and toks[0].endswith("com"):
            toks = [toks[0][:-3]]                     # indianinfrastructurecom
            was_site = True
        if was_site and self.vocab:                   # urbanlakshmiservices -> urban lakshmi services
            toks = [p for t in toks for p in (self._segment(t) if t not in self.vocab else (t,))]
        toks = self._merge_single_letters(toks)
        toks = [self._canon_name_token(t) for t in toks]
        toks = [t for t in toks if t != "www"]
        if was_indic:                                 # words the learned dictionary never saw
            toks = _indic_legal_fallback(toks)
        toks = ["private" if t == "p" and i + 1 < len(toks) and toks[i + 1] == "limited" else t
                for i, t in enumerate(toks)]
        while len(toks) > 1 and toks[0] in HONORIFIC:
            toks = toks[1:]
        toks = _dedupe_consecutive(toks)
        legal = sorted({t for t in toks if t in LEGAL})
        core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
        if not core:
            core = [t for t in toks if t not in NAME_STOP] or toks
        core = _dedupe_consecutive(core)
        bare = _BARE_COUNTRY_WORDS.get(normalize_country(country)) if country else None
        if bare and len(core) > 1:                    # bare "France" = "(France)", which is dropped
            core = [t for t in core if t not in bare] or core
        key = [t for t in core if t not in NAME_FILLER] or core
        return {
            "name_norm": " ".join(toks),
            "name_core": " ".join(core),
            "name_key": " ".join(sorted(key)),
            "name_compact": "".join(core),
            "name_tokens": tuple(core),
            "name_initials": "".join(t[0] for t in core if not t.isdigit()),
            "name_legal": " ".join(legal),
            "name_n_digits": sum(ch.isdigit() for ch in "".join(core)),
            "name_was_indic": int(was_indic),
            "name_had_alias": had_alias,
        }

    def _segment(self, t: str) -> tuple:
        """Split a glued website name into Source 1 words (most likely split by word counts).
        Only if EVERY piece is a known word of 2+ letters and there are 2+ pieces; else unchanged."""
        hit = self._seg_cache.get(t)
        if hit is not None:
            return hit
        res = (t,)
        n = len(t)
        if self.vocab and 6 <= n <= 60 and t.isalpha():
            inf = float("inf")
            best, back = [0.0] + [inf] * n, [0] * (n + 1)
            for e in range(2, n + 1):
                for b in range(max(0, e - 20), e - 1):
                    if best[b] == inf:
                        continue
                    c = self.vocab.get(t[b:e])
                    if c:
                        cost = best[b] + self._vlog_n - math.log(c)
                        if cost < best[e]:
                            best[e], back[e] = cost, b
            if best[n] < inf:
                pieces, e = [], n
                while e > 0:
                    pieces.append(t[back[e]:e])
                    e = back[e]
                if len(pieces) >= 2:
                    res = tuple(reversed(pieces))
        if len(self._seg_cache) < CACHE_LIMITS["name"]:
            self._seg_cache[t] = res
        return res

    @staticmethod
    def _merge_single_letters(toks: list[str]) -> list[str]:
        """l l c -> llc, s k enterprises -> sk enterprises."""
        out, run = [], []
        for t in toks + [""]:
            if len(t) == 1 and t.isalpha():
                run.append(t)
                continue
            if len(run) >= 2:
                out.append("".join(run))
            else:
                out.extend(run)
            run = []
            if t:
                out.append(t)
        return out

    # ---------------------------------------------------------------- addresses
    def _component(self, comp: str, fr: bool = False) -> tuple:
        """one comma-separated component -> (clean key for state lookup, canonical tokens)."""
        ck = (comp, fr)
        hit = self._comp_cache.get(ck)
        if hit is not None:
            return hit
        c = base_clean(comp)
        key = _WS_RE.sub(" ", c.replace("-", " ")).strip()
        raw = tokenize(c)
        can = [self._canon_addr_token(t) for t in raw]
        for i in range(1, len(can)):
            after_num = bool(_HOUSE_NUM_RE.match(can[i - 1])) or (
                i >= 2 and _HOUSE_NUM_RE.match(can[i - 2])
                and (can[i - 1] in HOUSE_SFX or (len(can[i - 1]) == 1 and can[i - 1].isalpha())))
            if not after_num:
                continue
            nxt = can[i + 1] if i + 1 < len(can) else ""
            if fr:                                     # 41 avneue -> 41 avenue
                can[i] = _fix_street(can[i])
                nxt = _fix_street(nxt) if nxt else nxt
            if raw[i] == "ter" and nxt in STREET_TYPES:  # "7 ter rue" is 7-ter, not 7 terrace
                can[i] = "ter"
        if fr:                                         # the fix above may have changed a later word
            for i in range(2, len(can)):
                if can[i - 1] == "ter" and _HOUSE_NUM_RE.match(can[i - 2]):
                    can[i] = _fix_street(can[i])
        toks = tuple(t for t in can if t not in ADDR_DROP and t not in ADDR_STOP)
        res = (key, toks)
        if len(self._comp_cache) < CACHE_LIMITS["comp"]:
            self._comp_cache[ck] = res
        return res

    def addr_repr(self, raw: str, country: str = "") -> dict:
        raw = raw or ""
        fr = bool(country) and normalize_country(country) == "fr"
        s, _, _ = self.translit.to_latin(raw, states=True)
        s = s.translate(_PUNCT_MAP)
        s = _A_NULL_RE.sub(",", s)
        s = _A_POBOX_RE.sub(" pobox ", s)
        s = _A_HNO_RE.sub(" no ", s)
        s = _A_LAYOUT_RE.sub(" layout ", s)
        s = _A_CARE_RE.sub(" ", s)
        s = s.replace("#", " ").replace("(", " ").replace(")", " ").replace(";", ",")
        keys, comps = [], []
        for part in s.split(","):
            if not part.strip():
                continue
            key, toks = self._component(part.strip(), fr)
            if toks or key:
                keys.append(key)
                comps.append(list(toks))

        # 1) POSTAL CODE FIRST, so "CA 95113" / "Karnataka 560001" / "75001 Paris" leave a clean
        #    state or city behind.  Searched from the end: postcodes sit at the end of an address.
        postal = ""
        for ci in range(len(comps) - 1, -1, -1):
            c = comps[ci]
            if "pobox" in c:
                continue
            hit = _find_postal(c)
            if hit is None:
                continue
            postal, del_idx = hit
            for d in sorted(del_idx, reverse=True):
                del c[d]
            postal = postal.split("-")[0]
            keys[ci] = _WS_RE.sub(" ", re.sub(r"\b\d{3}\s?\d{3}\b|\b\d{5,6}(?:-\d{4})?\b", " ",
                                                keys[ci])).strip(" -")
            break

        # 2) STATE / REGION: last component that IS a state (after the postcode was cut off)
        state, pos = "", -1
        for ci, key in enumerate(keys):
            code = ADMIN.get(key)
            if code:                                      # "DE" has no tokens left but is Delaware
                state, pos = code, ci
        if pos >= 0:
            comps[pos] = []                               # earlier ones ("New York, New York") stay text
        else:
            for c in reversed(comps):                     # "san jose ca" -> "san jose" + ca
                if len(c) >= 2 and c[-1] in STATE_CODES and not c[-2].isdigit():
                    state = c.pop()
                    break
                n = _trailing_state_len(c)                # "bangalore karnataka" -> "bangalore" + ka
                if n:
                    state = ADMIN[" ".join(c[-n:])]
                    del c[-n:]
                    break
        comps = [c for c in comps if c]
        house, house_sfx, street = self._house_street(comps)
        tokens = [t for c in comps for t in c]
        return {
            "addr_norm": " ".join(tokens),
            "addr_tokens": tuple(tokens),
            "addr_alpha": tuple(t for t in tokens if not any(ch.isdigit() for ch in t)),
            "addr_nums": tuple(t for t in tokens if any(ch.isdigit() for ch in t)),
            "addr_comps": tuple(sorted({" ".join(c) for c in comps})),
            "addr_house": house,
            "addr_house_sfx": house_sfx,
            "addr_postal": postal,
            "addr_street": street,
            "addr_state": state,
            "addr_empty": int(not tokens and not state and not postal),
        }

    @staticmethod
    def _house_street(comps: list[list[str]]) -> tuple[str, str, str]:
        num_re = re.compile(r"^[a-z]?\d[0-9a-z/-]*$")
        for c in comps:  # 0) "plot 5 ..." / "pl 5 ..." -> 5
            if len(c) >= 2 and c[0] in ("pl", "plot", "house", "door", "khasra") and num_re.match(c[1]):
                return c[1].lstrip("0") or "0", "", " ".join(c[2:])
        for c in comps:  # 1) a component that starts with a number and is not a unit/floor/box
            if (c and num_re.match(c[0]) and not (len(c) > 1 and c[1] in UNIT_WORDS)
                    and "floor" not in c and "pobox" not in c):
                rest = c[1:]
                sfx = ""
                if rest and (rest[0] in HOUSE_SFX or (len(rest[0]) == 1 and rest[0].isalpha()
                                                      and len(rest) > 1 and rest[1] in STREET_TYPES)):
                    sfx, rest = rest[0], rest[1:]
                h = c[0]
                m = re.match(r"^(\d+)(?:-?([a-z])|-\d+)$", h)
                if m:                                  # 12a / 2418-b -> 12 + a ; 4003-4007 -> 4003
                    h, sfx = m.group(1), sfx or (m.group(2) or "")
                h = h.lstrip("0") or "0"
                return h, sfx, " ".join(rest)
        for c in comps:  # 2) number right before a street-type word
            for k in range(len(c) - 1):
                if num_re.match(c[k]) and any(t in STREET_TYPES for t in c[k + 1:k + 4]):
                    return c[k].lstrip("0") or "0", "", " ".join(c[k + 1:])
        for c in comps:  # 3) no number: first component naming a street
            if set(c) & STREET_TYPES:
                return "", "", " ".join(c)
        return "", "", ""

    # ---------------------------------------------------------------- record
    def record(self, name: str, addr: str, country: str) -> dict:
        r = self.name_repr(name, country)
        r.update(self.addr_repr(addr, country))
        r["country_norm"] = normalize_country(country)
        r["name_len"] = len(r["name_core"])
        r["addr_len"] = len(r["addr_norm"])
        r["name_ntok"] = len(r["name_tokens"])
        r["addr_ntok"] = len(r["addr_tokens"])
        return r

    # ---------------------------------------------------------------- training
    def fit(self, s1_rows, target_rows, owner: dict, max_abbrev_pairs: int = 300_000,
            abbrev_min_count: int = 25, protect_min_count: int = 5, log=print) -> "Normalizer":
        """
        s1_rows / target_rows: iterables of (entity_id, name, address, country).
        owner: {S2/S3 id: S1 id} from the ground truth.
        Step A: learn the Indic dictionary.  Step B: mine abbreviation maps with it applied.
        """
        t0 = time.time()
        s1 = {r[0]: (r[1], r[2], r[3]) for r in s1_rows}
        wc = Counter()
        for nm, _, _ in s1.values():
            if nm and not INDIC_RE.search(nm):
                wc.update(t for t in tokenize(base_clean(nm)) if t.isalpha() and len(t) >= 2)
        self.vocab = dict(x for x in wc.most_common(300_000) if x[1] >= 2)
        self._vlog_n = math.log(sum(self.vocab.values()) + 1) if self.vocab else 0.0
        pairs = [(s1[owner[r[0]]], (r[1], r[2], r[3])) for r in target_rows
                 if r[0] in owner and owner[r[0]] in s1]
        indic = [(a[0], b[0]) for a, b in pairs if INDIC_RE.search(b[0])]
        self.translit = LearnedDict().fit(indic)
        self.name_map, self.addr_map = {}, {}
        # A word one letter away from a legal/common word ("pirate" ~ "private") is a TYPO only if
        # true matches write the other word.  If true matches keep the same word, it is a real word:
        # protect it, so "Pirate Pizza" does not lose "pirate".  Learned from the true pairs.
        kept, fixed = Counter(), Counter()
        for a, b in pairs[:: max(1, len(pairs) // 500_000)]:
            A = set(tokenize(base_clean(a[0])))
            B = set(tokenize(base_clean(b[0])))
            for t in A:
                if len(t) >= 5 and t.isalpha() and t not in LEGAL:
                    f = _fix_typo(t)
                    if f != t:
                        if t in B:
                            kept[t] += 1
                        elif f in B:
                            fixed[t] += 1
        self.protect = {t for t, n in kept.items() if n >= protect_min_count and n >= fixed[t]} \
            | _TYPO_PROTECT_SEED
        self._reset_caches()
        log(f"vocabulary for website names: {len(self.vocab):,} Source 1 name words")
        log(f"translit: {len(self.translit.map)} words from {len(indic)} Indic pairs; "
            f"{len(self.protect)} protected look-alike words ({time.time() - t0:.1f}s)")
        step = max(1, len(pairs) // max_abbrev_pairs)
        sub = pairs[::step][:max_abbrev_pairs]
        nl, nr, al, ar = [], [], [], []
        for a, b in sub:
            nl.append(self.name_repr(a[0], a[2])["name_tokens"])
            nr.append(self.name_repr(b[0], b[2])["name_tokens"])
            al.append(self.addr_repr(a[1], a[2])["addr_tokens"])
            ar.append(self.addr_repr(b[1], b[2])["addr_tokens"])
        prot_n = set(NAME_ABBREV) | set(NAME_ABBREV.values()) | LEGAL | NAME_STOP | HONORIFIC
        prot_a = set(ADDR_ABBREV) | set(ADDR_ABBREV.values()) | ADDR_DROP | ADDR_STOP | STATE_CODES
        self.name_map = mine_abbreviations(nl, nr, abbrev_min_count, prot_n)
        self.addr_map = mine_abbreviations(al, ar, abbrev_min_count, prot_a)
        self._reset_caches()
        log(f"abbreviations: {len(self.name_map)} name, {len(self.addr_map)} address maps "
            f"from {len(sub)} pairs ({time.time() - t0:.1f}s)")
        return self

    def _reset_caches(self):
        self._seg_cache.clear()
        self._name_tok_cache.clear()
        self._addr_tok_cache.clear()
        self._comp_cache.clear()


# ---- helpers added in v2 -------------------------------------------------------------------
_KEEP_ZERO_RE = re.compile(r"^0[1-9]\d{3}$")          # 06000 (Nice), 04101 (Portland ME)
_POSTAL_RE = re.compile(r"^\d{5,6}(?:-\d{4})?$")
_TYPO_PROTECT_SEED = {"pirate", "pirates", "primate", "primates", "prate", "pilate"}
_TRAIL_STATES = {k for k in {**_US_STATES, **_IN_STATES}
                 if (" " in k or len(k) >= 6) and k not in ("washington", "delhi", "nct of delhi")}
_LEGAL_PK = {phonetic_key(w): w for w in ("private", "limited", "company", "corporation",
                                          "incorporated")}


def _find_postal(c: list[str]):
    """(postcode, token indices to delete) for one component, or None.
    Note: tokens are already canonical, so "700 001" arrives as ["700", "1"]."""
    if not c:
        return None
    if len(c) == 1:
        return (c[0], [0]) if _POSTAL_RE.match(c[0]) else None
    for k, t in enumerate(c):                            # 6 digits: Indian PIN, anywhere
        if len(t) == 6 and t.isdigit():
            return t, [k]
    k = len(c) - 2                                       # "kolkata 700 001" / "700 001"
    if (len(c[k]) == 3 and c[k].isdigit() and c[k][0] != "0" and c[k + 1].isdigit()
            and len(c[k + 1]) <= 3 and (k == 0 or not c[k - 1].isdigit())):
        return c[k] + c[k + 1].zfill(3), [k, k + 1]
    if _POSTAL_RE.match(c[-1]):
        return c[-1], [len(c) - 1]                       # "ca 95113", "portland 97477-1234"
    if (_POSTAL_RE.match(c[0]) and all(not any(ch.isdigit() for ch in t) for t in c[1:])
            and not set(c[1:]) & (STREET_TYPES | UNIT_WORDS)):
        return c[0], [0]                                 # French "75001 paris" (not "12345 main st")
    return None


def _trailing_state_len(c: list[str]) -> int:
    """How many trailing tokens of a component spell a (long, unambiguous) state name."""
    for n in (3, 2, 1):
        if len(c) > n and " ".join(c[-n:]) in _TRAIL_STATES:
            return n
    return 0


def _indic_legal_fallback(toks: list[str]) -> list[str]:
    """For names that came from an Indian script: 'pra li' (प्रा. लि.) -> private limited, and
    sound-alike spellings of legal words (praivet, limited, kampani) -> the English word."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t == "pra" and i + 1 < len(toks) and toks[i + 1] in ("li", "lim", "lmtd"):
            out += ["private", "limited"]
            i += 2
            continue
        if len(t) >= 5 and t not in LEGAL:
            t = _LEGAL_PK.get(phonetic_key(t), t)
        out.append(t)
        i += 1
    return out


def _dedupe_consecutive(toks: list[str]) -> list[str]:
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# =============================================================================
# Batch helpers (pipeline integration)
# =============================================================================
REPR_FIELDS = [
    "name_norm", "name_core", "name_key", "name_compact", "name_tokens", "name_initials", "name_legal",
    "name_n_digits", "name_was_indic", "name_had_alias",
    "addr_norm", "addr_tokens", "addr_alpha", "addr_nums", "addr_comps", "addr_house",
    "addr_house_sfx", "addr_postal", "addr_street", "addr_state", "addr_empty",
    "country_norm", "name_len", "addr_len", "name_ntok", "addr_ntok",
]


def _iter_rows(rows):
    """Accepts a pandas DataFrame (entity_id, business_name, business_address, country) or tuples."""
    if hasattr(rows, "itertuples"):
        yield from zip(rows["entity_id"], rows["business_name"], rows["business_address"], rows["country"])
    else:
        yield from rows


def build_repr(rows, norm: Normalizer) -> dict:
    """rows -> column dict {"entity_id", "business_name", "business_address", <REPR_FIELDS>}."""
    cols = {"entity_id": [], "business_name": [], "business_address": []}
    cols.update({f: [] for f in REPR_FIELDS})
    for eid, name, addr, country in _iter_rows(rows):
        r = norm.record(name, addr, country)
        cols["entity_id"].append(eid)
        cols["business_name"].append(name)
        cols["business_address"].append(addr)
        for f in REPR_FIELDS:
            cols[f].append(r[f])
    return cols


_INT_FIELDS = ("name_n_digits", "name_was_indic", "name_had_alias", "addr_empty",
               "name_len", "addr_len", "name_ntok", "addr_ntok")


def _to_frame(cols: dict):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(cols)
    for f in _INT_FIELDS:
        df[f] = df[f].astype(np.int32)
    return df


_WORKER: Normalizer | None = None


def _init_worker(state, cache_mode=None):
    global _WORKER
    if cache_mode:
        set_cache_limits(cache_mode)
    _WORKER = Normalizer.from_state(state)


def _work(chunk):
    return build_repr(chunk, _WORKER)


def repr_frame(df, norm: Normalizer, n_jobs: int = 4, chunk: int = 50_000, cache_mode=None):
    """DataFrame (entity_id, business_name, business_address, country) -> representation DataFrame,
    same row order, index 0..n-1.  This is what blocking.py and features.py consume.
    Uses a process pool for big inputs (on Windows call it under `if __name__ == "__main__":`)."""
    rows = list(_iter_rows(df))
    if n_jobs <= 1 or len(rows) < 2 * chunk:
        return _to_frame(build_repr(rows, norm))
    from multiprocessing import Pool
    parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(n_jobs, initializer=_init_worker, initargs=(norm.state(), cache_mode)) as pool:
        outs = pool.map(_work, parts)
    cols = {k: [] for k in outs[0]}
    for o in outs:
        for k, v in o.items():
            cols[k].extend(v)
    return _to_frame(cols)


parallel_repr = repr_frame   # old name


# =============================================================================
# CLI
# =============================================================================
def read_tsv(path: str):
    """Tab-separated, no quote handling, 'N/A' stays a string ."""
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        idx = [header.index(c) for c in ("entity_id", "business_name", "business_address", "country")]
        for row in r:
            row = row + [""] * (len(header) - len(row))
            yield tuple(row[i] for i in idx)


def read_truth(path: str) -> dict[str, set]:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            ids = row[1].split(",") if len(row) > 1 and row[1] else []
            out[row[0]] = {i.strip() for i in ids if i.strip()}
    return out


def _fmt(v):
    """For human-readable TSV only.  Tuples are joined with " | " so multi-word address
    components stay distinguishable ("12 mg road | bangalore")."""
    s = " | ".join(v) if isinstance(v, tuple) else str(v)
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _cmd_demo(args):
    norm = Normalizer.load(args.state) if args.state else Normalizer()
    names = [
        ("Acme Robotics Inc.", "US"), ("ACME ROBOTICS INCORPORATED", "US"),
        ("Solmira DBA Eastern (India) Mill Group", "India"), ("M/s Mehta (India) Sutra Pvt. Ltd", "India"),
        ("Shri Supreme Consulting Private  (Limited)", "India"), ("Rnd Techno1ogy Limited #29691", "India"),
        ("8rothers Gu1f LLC LLC", "US"), ("Harb0r Gulf Labs, PC | www.harbrgul.com", "US"),
        ("familystarhealth.com", "US"), ("श्याम कंसल्टिंग प्रा. लि.", "India"),
        ("ಮಾಡರ್ನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "India"), ("S.A.S. Connect Groupement France", "France"),
        ("Gagny (Frànce) Club (S.A.R.L.)", "France"), ("Hotesses & Frères SASU", "France"),
        ("SHYAM CARE PRISVATE-LIMITED", "India"), ("Kp Kp Energy Ltd (ID: 51109)", "India"),
    ]
    for n, c in names:
        r = norm.name_repr(n, c)
        print(f"{n!r:48} -> core={r['name_core']!r} legal={r['name_legal']} indic={r['name_was_indic']}")
    addrs = [
        "500 Market St, San Jose, CA", "San Jose CA, 500 MARKET STREET",
        "IA, Iowa City, 1064 Newton Rd, Unit 11", "2702 WATTS AVE, PORTSMOUTH, VA, <NULL>",
        "West Bengal, NO 22 PRINCE ANWAR SHAH ROAD, MERLIN OXFORD, KOLKATA",
        "H.NO 99 SECTOR A POCKET C, NEW DELHI, दिल्ली", "Mumbai City, Andheri East, Maharashtra",
        "GREATER BOMBAY, EKTA MEDOWS, BLDG NO.1, MH",
        "53 Bis Rue Firmin Colas, Nantes, Pays de la Loire", "N° 50 R. DE LA BENAUGE, BORDEAUX, Gironde",
        "00109 BOULEVARD DES BELGES, 44000, NANTES, Loire-Atlantique",
        "9 Ch De La Huossiniere, Bp 72616, Nantes, Loire-Atlantique", "ST.-NAZAIRE, 1 rue Jodelle",
        "New York, New York",
    ]
    for a in addrs:
        r = norm.addr_repr(a)
        print(f"{a!r:66} -> comps={r['addr_comps']} house={r['addr_house']!r}"
              f"{'+' + r['addr_house_sfx'] if r['addr_house_sfx'] else ''} street={r['addr_street']!r} "
              f"state={r['addr_state']!r} postal={r['addr_postal']!r}")


def fit_from_files(data_dir: str, sample_mod: int = 1, log=print) -> Normalizer:
    """Fit a Normalizer on dataset/train (sample_mod k keeps S1 entities with crc32(id) % k == 0)."""
    import os
    d = os.path.join(data_dir, "train")
    keep = (lambda i: zlib.crc32(i.encode()) % sample_mod == 0) if sample_mod > 1 else (lambda i: True)
    truth = read_truth(os.path.join(d, "train_ground_truth.tsv"))
    owner = {t: s for s, ts in truth.items() if keep(s) for t in ts}
    s1 = [r for r in read_tsv(os.path.join(d, "train_source1.tsv")) if keep(r[0])]
    tg = [r for src in ("2", "3") for r in read_tsv(os.path.join(d, f"train_source{src}.tsv"))
          if r[0] in owner]
    log(f"fit on {len(s1):,} S1 and {len(tg):,} matched S2/S3 records")
    return Normalizer().fit(s1, tg, owner, log=log)


def heldout_report(data_dir: str, report_mod: int = 8, log=print):
    """Fit on 80% of (a 1/report_mod sample of) S1 entities, measure on the other 20%:
    how often a TRUE pair ends up with identical cleaned fields, before vs after learning."""
    import os
    d = os.path.join(data_dir, "train")
    keep = (lambda i: zlib.crc32(i.encode()) % report_mod == 0) if report_mod > 1 else (lambda i: True)
    truth = read_truth(os.path.join(d, "train_ground_truth.tsv"))
    owner = {t: s for s, ts in truth.items() if keep(s) for t in ts}
    s1 = [r for r in read_tsv(os.path.join(d, "train_source1.tsv")) if keep(r[0])]
    tg = [r for src in ("2", "3") for r in read_tsv(os.path.join(d, f"train_source{src}.tsv"))
          if r[0] in owner]
    hold = lambda i: zlib.crc32(i.encode()) % 5 == 0
    n_tr = Normalizer().fit([r for r in s1 if not hold(r[0])],
                            [r for r in tg if not hold(owner[r[0]])], owner, log=log)
    base = Normalizer()
    s1d = {r[0]: r for r in s1}
    stats = defaultdict(Counter)
    for r in tg:
        o = owner[r[0]]
        if not hold(o) or o not in s1d:
            continue
        a = s1d[o]
        seg = ("indic" if INDIC_RE.search(r[1]) else normalize_country(a[3]))
        for tag, nm in (("before", base), ("after", n_tr)):
            x, y = nm.name_repr(a[1], a[3]), nm.name_repr(r[1], r[3])
            stats[seg][tag + "_core_eq"] += x["name_core"] == y["name_core"]
            stats[seg][tag + "_key_eq"] += x["name_key"] == y["name_key"]
            stats[seg][tag + "_tok_overlap"] += bool(set(x["name_tokens"]) & set(y["name_tokens"]))
        xa, ya = n_tr.addr_repr(a[2], a[3]), n_tr.addr_repr(r[2], r[3])
        for fld, tag in (("addr_state", "state"), ("addr_house", "house"), ("addr_postal", "postal")):
            if xa[fld] and ya[fld]:
                stats[seg][tag + "_both"] += 1
                stats[seg][tag + "_eq"] += xa[fld] == ya[fld]
        stats[seg]["n"] += 1
    log("\nheld-out TRUE pairs (maps learned on the other 80% of entities). Higher = better.")
    log(f"{'segment':8} {'pairs':>8} {'core== before':>14} {'core== after':>13} {'key== after':>12} "
        f"{'shared tok':>11} {'state==':>8} {'house==':>8} {'postal==':>9}")
    for seg, c in sorted(stats.items()):
        n = max(1, c["n"])
        log(f"{seg:8} {c['n']:8d} {c['before_core_eq'] / n:14.3f} {c['after_core_eq'] / n:13.3f} "
            f"{c['after_key_eq'] / n:12.3f} {c['after_tok_overlap'] / n:11.3f} "
            f"{c['state_eq'] / max(1, c['state_both']):8.3f} {c['house_eq'] / max(1, c['house_both']):8.3f} "
            f"{c['postal_eq'] / max(1, c['postal_both']):9.3f}")
    # precision proxy: S1 is deduplicated, so S1 records sharing a key are distinct businesses
    for fld in ("name_core", "name_key"):
        grp = Counter((normalize_country(r[3]), n_tr.name_repr(r[1], r[3])[fld]) for r in s1)
        shared = sum(v for v in grp.values() if v > 1)
        log(f"S1 records whose {fld} is shared with another S1 record: {shared / max(1, len(s1)):.3f} "
            f"(these names alone can never decide a match)")


def _cmd_fit(args):
    """Optional held-out report, then fit on (all of) train and save."""
    import os
    t0 = time.time()
    if not args.no_report:
        heldout_report(args.data_dir, args.report_mod)
    norm = fit_from_files(args.data_dir, args.sample_mod)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    norm.save(args.out)
    print(f"\nsaved {args.out}  (total {time.time() - t0:.0f}s)")


def _cmd_apply(args):
    norm = Normalizer.load(args.state) if args.state else Normalizer()
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(["entity_id"] + REPR_FIELDS)
        for eid, name, addr, country in read_tsv(args.inp):
            r = norm.record(name, addr, country)
            w.writerow([eid] + [_fmt(r[k]) for k in REPR_FIELDS])  # tuples: " | "-joined
            n += 1
    print(f"wrote {n} rows to {args.out} ({time.time() - t0:.0f}s)")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("demo", help="print normalised examples")
    a.add_argument("--state", default=None)
    a = sp.add_parser("fit", help="learn Indic dictionary + abbreviation maps from train")
    a.add_argument("--data-dir", required=True, help="folder containing train/")
    a.add_argument("--out", default="work/norm_state.json")
    a.add_argument("--sample-mod", type=int, default=1,
                   help="final fit on S1 entities with crc32(id) %% k == 0 (1 = all, recommended)")
    a.add_argument("--report-mod", type=int, default=8, help="sample used by the held-out report")
    a.add_argument("--no-report", action="store_true", help="skip the held-out report (faster)")
    a = sp.add_parser("apply", help="normalise one source TSV")
    a.add_argument("--state", default=None)
    a.add_argument("--inp", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"demo": _cmd_demo, "fit": _cmd_fit, "apply": _cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
