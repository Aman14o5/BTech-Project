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


# ---------------------------------------------------------------- evidence contradiction
"""
Evidence contradiction: replace ONE evidence word with content that conflicts with it,
so the sentence stays grammatical (unlike removal) and the label should follow the edit.

  contradict_cited    edit one word the model's explanation cites
  contradict_random   control: the same kind of edit on a non-cited content word
  contradict_gold     edit one human-highlighted word (model-independent)

Each edit is one of three operations, tried in this order of priority:
  align        the word clashes with a word in the other sentence (premise "pool" vs
               hypothesis "ocean"); copy the other sentence's word in ("pool" -> "ocean").
               Typically turns a contradiction into an entailment.
  swap_shared  the word also occurs in the other sentence ("man" in both); replace it with an
               incompatible word ("woman"). Typically turns an entailment into a contradiction.
  swap         any other evidence word; replace it with an incompatible word.

The expected label of every edited pair is NOT decided by rules: an independent NLI
model (the oracle) labels it, and only confident oracle labels are scored. Faithful
explanation => editing the cited word moves the model to the oracle's new label, and
does so more often than editing a non-cited word.
"""
CONTRADICTION_OPS = ("align", "swap_shared", "swap")


def _contradiction_options(tokens, indices, other_tokens, swapper):
    """{op: [(index, [substitutes])]} for the words at `indices`."""
    this_words = {normalise(t) for t in tokens}
    other_words = {normalise(t) for t in other_tokens}
    other_content = [other_tokens[j] for j in content_indices(" ".join(other_tokens))
                     if normalise(other_tokens[j]) not in this_words]
    opts = {op: [] for op in CONTRADICTION_OPS}
    for i in indices:
        tok = tokens[i]
        partners = [swapper.transplant(tok, o) for o in other_content if swapper.related(tok, o)]
        if partners:
            opts["align"].append((i, sorted(set(partners))))
            continue
        subs = [s for s in swapper.substitutes(tok)
                if normalise(s) not in this_words and normalise(s) not in other_words]
        if subs:
            opts["swap_shared" if normalise(tok) in other_words else "swap"].append((i, subs))
    return opts


def _pick(opts, rng, prefer_op=None):
    order = ([prefer_op] if prefer_op else []) + [op for op in CONTRADICTION_OPS if op != prefer_op]
    for op in order:
        if opts.get(op):
            i, subs = opts[op][rng.randrange(len(opts[op]))]
            return op, i, subs[rng.randrange(len(subs))]
    return None


def build_contradiction_variants(example, cited, swapper, side="premise", seed=0, include_gold=True):
    """Return {"<side>:contradict_<x>": edit} where edit = {"premise", "hypothesis", "op",
    "index", "target", "substitute"}. Edits that cannot be built are omitted."""
    rng = random.Random(f"{seed}-{example['id']}-{side}-contradict")
    other = "hypothesis" if side == "premise" else "premise"
    tokens, other_tokens = example[side].split(), example[other].split()
    content = set(content_indices(example[side]))

    def edit(choice):
        if choice is None:
            return None
        op, i, sub = choice
        new = list(tokens)
        new[i] = sub
        if i > 0 and new[i - 1].lower() in ("a", "an"):  # "a pool" -> "an ocean"
            art = "an" if sub.lstrip("\"'(").lower()[:1] in tuple("aeiou") else "a"
            new[i - 1] = art.capitalize() if new[i - 1][0].isupper() else art
        new_sent = " ".join(new)
        p, h = (new_sent, example["hypothesis"]) if side == "premise" else (example["premise"], new_sent)
        return {"premise": p, "hypothesis": h, "op": op, "index": i, "target": tokens[i], "substitute": sub}

    v = {}
    cited = [i for i in cited if i in content]
    if cited:
        c = _pick(_contradiction_options(tokens, cited, other_tokens, swapper), rng)
        v["contradict_cited"] = edit(c)
        if c:
            # matched control: same operation if possible, on a word the explanation does not cite
            rest = sorted(content - set(cited))
            v["contradict_random"] = edit(_pick(_contradiction_options(tokens, rest, other_tokens, swapper),
                                                rng, prefer_op=c[0]))
    gold = [i for i in ((example.get(f"gold_{side}_idx") or []) if include_gold else []) if i in content]
    if gold:
        v["contradict_gold"] = edit(_pick(_contradiction_options(tokens, gold, other_tokens, swapper), rng))
    return {f"{side}:{k}": e for k, e in v.items() if e is not None}
