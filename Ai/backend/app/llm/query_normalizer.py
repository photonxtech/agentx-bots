import difflib
import re

from spellchecker import SpellChecker

from app.llm.bm25_index import site_vocabulary

# One shared dictionary instance — it's read-only per lookup, so there's no reason to
# rebuild it per request or per website.
_SPELL = SpellChecker()

_WORD_RE = re.compile(r"[A-Za-z]+")

# How close a word must be (0-1, from difflib's ratio) to a real site-vocabulary word to
# treat it as a typo of that word rather than leaving it alone — high enough to avoid
# conflating genuinely different words, low enough to catch multi-letter typos like
# "priceings" -> "pricing" that a generic dictionary's edit-distance search can miss
# entirely (it only considers words reachable within a fixed edit distance, and "pricing"
# turned out not to be one of them for that particular misspelling).
_FUZZY_CUTOFF = 0.75

# Stricter than _FUZZY_CUTOFF: merging two words into one is a more disruptive edit than
# correcting a single word, so it needs stronger evidence — paired with the length-
# closeness check in _merge_split_words, this keeps merging to genuine split-word typos.
_MERGE_CUTOFF = 0.85


def _merge_split_words(question: str, vocab: set[str]) -> str:
    # A stray space (or a nearby key hit while typing it) can split one intended word into
    # two valid-looking ones ("plan form" for "platform", itself off by a letter from the
    # naive concatenation "planform") — neither half looks like a typo on its own, so the
    # per-word pass below never touches them. Fuzzy-match the concatenation against the
    # site's own vocabulary — a good match there is a strong enough signal to merge, since
    # it's specifically a real term used on this site.
    words = question.split(" ")
    merged: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            core1 = re.sub(r"[^A-Za-z]", "", words[i]).lower()
            core2 = re.sub(r"[^A-Za-z]", "", words[i + 1]).lower()
            concat = core1 + core2
            if core1 and core2:
                # difflib's ratio counts matched characters, not whole-word similarity, so
                # a short vocab word (e.g. "ceo") can score deceptively high just for being
                # a substring of an unrelated concatenation ("isceo"). Requiring the
                # matched word's length to be close to the concatenation's rules that out —
                # a genuine split-word typo doesn't change the total letter count by much.
                close = difflib.get_close_matches(concat, vocab, n=1, cutoff=_MERGE_CUTOFF)
                if close and abs(len(close[0]) - len(concat)) <= 1:
                    merged.append(close[0])
                    i += 2
                    continue
        merged.append(words[i])
        i += 1
    return " ".join(merged)


def correct_query_spelling(website_id: int, question: str) -> str:
    # Only correct words that are: lowercase (skip likely proper nouns/acronyms typed with
    # capitals), at least 3 letters, not already a real dictionary word, AND not part of the
    # site's own vocabulary (brand names, abbreviations like "ceo" aren't English dictionary
    # words but are correct as-is for this site — the whitelist prevents e.g. "ceo" -> "coo"
    # or "photonx" -> "photon").
    vocab = site_vocabulary(website_id)
    question = _merge_split_words(question, vocab)

    def _fix(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 3 or not word.islower():
            return word
        if word in vocab or word in _SPELL:
            return word

        # Try the site's own vocabulary first — it's a small, targeted search space, so a
        # fuzzy match against it is both more likely to succeed and more likely to be the
        # actually-intended word than a generic English dictionary correction.
        close = difflib.get_close_matches(word, vocab, n=1, cutoff=_FUZZY_CUTOFF)
        if close:
            return close[0]

        suggestion = _SPELL.correction(word)
        if not suggestion or suggestion == word:
            return word
        return suggestion

    return _WORD_RE.sub(_fix, question)
