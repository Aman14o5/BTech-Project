"""Identify which words of the input an explanation cites.

A word of the premise/hypothesis counts as *cited* when its normalised form
(lower-cased, punctuation stripped, light stemming) appears in the model's
explanation and it is not a stopword. This is simple and transparent, which is
what we want for a first version; an LLM- or embedding-based aligner can be
swapped in later behind the same function signature.
"""
import re

STOPWORDS = set("""
a an the and or but if then than so as of at by for with about against between into
through during before after above below to from up down in out on off over under again
further once here there when where why how all any both each few more most other some such
no nor not only own same too very can will just don should now is are was were be been being
have has had having do does did doing i me my myself we our ours you your he him his she her
it its they them their what which who whom this that these those am would could ought
because while until s t also just there's it's
""".split())

_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def normalise(word):
    w = _PUNCT.sub("", word.lower())
    for suf in ("ing", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_indices(sentence):
    """Indices of non-stopword, non-empty tokens in `sentence.split()`."""
    out = []
    for i, tok in enumerate(sentence.split()):
        n = normalise(tok)
        if n and n not in STOPWORDS:
            out.append(i)
    return out


def cited_indices(sentence, explanation):
    """Indices of tokens in `sentence` whose normalised form occurs in `explanation`."""
    expl = {normalise(t) for t in explanation.split()}
    expl.discard("")
    return [i for i in content_indices(sentence) if normalise(sentence.split()[i]) in expl]


def overlap_prf(pred, gold):
    """Token-set precision/recall/F1 between cited and gold-highlighted indices."""
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return None
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}
