"""Counterfactual interventions on the evidence an explanation cites.

Each intervention edits ONE side of the pair and leaves the other untouched.
Variant keys are "<side>:<name>", e.g. "premise:remove_cited".

  remove_cited          delete the words the model's explanation cites
  remove_random         control: delete the same number of other content words
  remove_overlap        stricter control: delete the same number of non-cited words
                        that ALSO occur in the other sentence (matches the lexical-
                        overlap property of cited words, which random words lack)
  isolate_cited         keep only the cited words                (premise side only)
  isolate_random        control: keep the same number of other content words
  remove_gold / isolate_gold / remove_gold_random
                        the same, with human-highlighted words (model-independent)

Expected behaviour if an explanation is faithful:
  removing cited evidence changes the prediction MORE than removing control words;
  keeping only the cited evidence preserves the prediction MORE than keeping control words.
"""
import random

from evidence import content_indices, normalise


def _drop(tokens, idx):
    idx = set(idx)
    return " ".join(t for i, t in enumerate(tokens) if i not in idx)


def _keep(tokens, idx):
    idx = set(idx)
    return " ".join(t for i, t in enumerate(tokens) if i in idx)


def _sample_like(pool, k, rng, fallback=()):
    """k indices from `pool` (topped up from `fallback`), or None if impossible."""
    pool, fallback = list(pool), [i for i in fallback if i not in set(pool)]
    if k == 0:
        return None
    if len(pool) >= k:
        return sorted(rng.sample(pool, k))
    if len(pool) + len(fallback) >= k:
        return sorted(pool + rng.sample(fallback, k - len(pool)))
    return None


def build_variants(example, cited, side="premise", seed=0, include_gold=True):
    """Return {"<side>:<variant>": (premise, hypothesis)} for one example.

    Variants that cannot be built (no cited words, no room for a matched
    control, empty result) are omitted, so every metric reports its own n.
    """
    rng = random.Random(f"{seed}-{example['id']}-{side}")
    other = "hypothesis" if side == "premise" else "premise"
    sent = example[side]
    tokens = sent.split()
    other_words = {normalise(w) for w in example[other].split()}
    content = content_indices(sent)
    gold = (example.get(f"gold_{side}_idx") or []) if include_gold else []

    def pair(new_sent):
        if not new_sent.strip():
            return None
        return (new_sent, example["hypothesis"]) if side == "premise" else (example["premise"], new_sent)

    def not_in(idx):
        s = set(idx)
        return [i for i in content if i not in s]

    allidx = range(len(tokens))
    v = {}
    if cited:
        k = len(cited)
        v["remove_cited"] = pair(_drop(tokens, cited))
        rand = _sample_like(not_in(cited), k, rng, [i for i in allidx if i not in set(cited)])
        if rand:
            v["remove_random"] = pair(_drop(tokens, rand))
        overlap = [i for i in not_in(cited) if normalise(tokens[i]) in other_words]
        ov = _sample_like(overlap, k, rng)  # no fallback: must be overlap words
        if ov:
            v["remove_overlap"] = pair(_drop(tokens, ov))
        if side == "premise":
            v["isolate_cited"] = pair(_keep(tokens, cited))
            if rand:
                v["isolate_random"] = pair(_keep(tokens, rand))
    if gold:
        v["remove_gold"] = pair(_drop(tokens, gold))
        grand = _sample_like(not_in(gold), len(gold), rng, [i for i in allidx if i not in set(gold)])
        if grand:
            v["remove_gold_random"] = pair(_drop(tokens, grand))
        if side == "premise":
            v["isolate_gold"] = pair(_keep(tokens, gold))
    return {f"{side}:{k}": p for k, p in v.items() if p is not None}
