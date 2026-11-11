"""Counterfactual interventions on the evidence an explanation cites.

All interventions edit one side of the pair (the premise by default) and leave
the other side untouched, so the claim being judged stays the same.

  remove_cited     delete the cited words                -> faithful: prediction should change
  remove_random    delete the same number of other words -> control: should change less
  isolate_cited    keep only the cited words             -> faithful: prediction should stay
  isolate_random   keep the same number of other words   -> control: should stay less often
  remove_gold / isolate_gold   same, using human-highlighted words (model-independent)

`remove_random` / `isolate_random` sample from non-cited *content* words first
(falling back to any non-cited word), so the control is not trivially weaker
just because it deletes stopwords.
"""
import random

from evidence import content_indices


def _drop(tokens, idx):
    idx = set(idx)
    return " ".join(t for i, t in enumerate(tokens) if i not in idx)


def _keep(tokens, idx):
    idx = set(idx)
    return " ".join(t for i, t in enumerate(tokens) if i in idx)


def _random_like(sentence, cited, rng):
    tokens = sentence.split()
    k = len(cited)
    content = [i for i in content_indices(sentence) if i not in set(cited)]
    other = [i for i in range(len(tokens)) if i not in set(cited) and i not in set(content)]
    if len(content) >= k:
        return sorted(rng.sample(content, k))
    if len(content) + len(other) >= k:
        return sorted(content + rng.sample(other, k - len(content)))
    return None  # not enough words to build a matched control


def build_variants(example, cited, side="premise", seed=0):
    """Return {variant_name: (premise, hypothesis)} for one example.

    `cited` are the model-cited indices on `side`. Variants that cannot be
    built (no cited words, no room for a control, empty result) are omitted.
    """
    rng = random.Random(f"{seed}-{example['id']}")
    sent = example[side]
    tokens = sent.split()
    gold = example[f"gold_{side}_idx"]

    def pair(new_sent):
        if not new_sent.strip():
            return None
        return (new_sent, example["hypothesis"]) if side == "premise" else (example["premise"], new_sent)

    variants = {}
    if cited:
        variants["remove_cited"] = pair(_drop(tokens, cited))
        variants["isolate_cited"] = pair(_keep(tokens, cited))
        rand = _random_like(sent, cited, rng)
        if rand is not None:
            variants["remove_random"] = pair(_drop(tokens, rand))
            variants["isolate_random"] = pair(_keep(tokens, rand))
    if gold:
        variants["remove_gold"] = pair(_drop(tokens, gold))
        variants["isolate_gold"] = pair(_keep(tokens, gold))
    return {k: v for k, v in variants.items() if v is not None}
