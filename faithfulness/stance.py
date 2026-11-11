"""Rule-based stance of a free-text NLI explanation: which label does the
explanation *argue for*, independent of the label the model predicted?

  "the fact that X doesn't imply Y", "just because", "could be"  -> neutral
  "X cannot Y", "X is not Y", "different"                        -> contradiction
  restatement / paraphrase ("X is Y", "another way of saying")   -> entailment

Comparing this stance with the predicted label measures label-explanation
consistency: an explanation that argues "doesn't imply" for an *entailment*
prediction cannot be the reason for that prediction.

The rules are validated on the human-written e-SNLI explanations (gold label
vs. stance of the human explanation); see `validate` and build_dataset.py.
"""
import re

_NEUTRAL_STRONG = [
    r"\b(does|do|did) ?(not|n't) (necessarily )?(imply|mean|indicate|say|state|show)",
    r"\bnot necessarily\b", r"\bjust because\b", r"\bnot all\b", r"\bnot every\b",
    r"\bno (indication|evidence|mention)\b", r"\bnot (stated|mentioned|specified|known|clear|sure)\b",
    r"\bdoesn'?t (say|mention|specify|tell)\b", r"\bunknown\b", r"\bcan'?t (be sure|tell|know)\b",
    r"\bisn'?t (stated|mentioned|clear)\b", r"\bnot (all|always)\b", r"\bdoes not have to\b",
    r"\bdoesn'?t have to\b", r"\bnot (indicated|implied)\b", r"\bno way (of|to) know",
    r"\bcan ?(not|n't) (infer|assume|tell|know|conclude|be sure)", r"\bnot assumed\b",
    r"\b(may|might) ?(not|n't)\b", r"\b(do|does|need) ?(not|n't) need to\b", r"\bneed ?(not|n't)\b",
    r"\b(are|is|was|were) ?(not|n't) (always|necessarily|only)\b", r"\b(isn|aren)'t (always|necessarily|only)\b",
    r"\bnot the only\b", r"\bno explanation\b", r"\bnot only\b", r"\bin many ways\b",
    r"\bmany (different|other) ", r"\bnot (said|told|given)\b",
]
_CONTRADICTION = [
    r"\bcannot\b", r"\bcan'?t\b", r"\bcan not\b", r"\bcould ?(not|n't)\b",
    r"\b(is|are|was|were|am) ?(not|n't)\b", r"\b(isn|aren|wasn|weren)'?t\b",
    r"\bnot\b", r"\bno\b", r"\bnobody\b", r"\bnever\b", r"\bdifferent\b", r"\bopposite\b",
    r"\bcontradict", r"\bimpossible\b", r"\bcan only\b", r"\bunless\b",
    r"\beither\b", r"\bwould ?(not|n't)\b", r"\bwhile\b", r"\bas opposed to\b", r"\bor\b.*\?",
    r"\binstead\b", r"\brather than\b", r"\bnot both\b",
]
_NEUTRAL_WEAK = [r"\bcould\b", r"\bmight\b", r"\bmay\b", r"\bpossibl", r"\bperhaps\b", r"\bmaybe\b",
                 r"\bnecessarily\b", r"\bassum"]

_compile = lambda ps: [re.compile(p) for p in ps]
_NS, _C, _NW = _compile(_NEUTRAL_STRONG), _compile(_CONTRADICTION), _compile(_NEUTRAL_WEAK)


def stance(explanation):
    e = " ".join(str(explanation or "").lower().replace("’", "'").split())
    if not e:
        return "none"
    if any(p.search(e) for p in _NS):
        return "neutral"
    if any(p.search(e) for p in _C):
        return "contradiction"
    if any(p.search(e) for p in _NW):
        return "neutral"
    return "entailment"


def validate(examples):
    """Accuracy of `stance` on human explanations vs gold labels, plus a confusion dict."""
    labels = ["entailment", "neutral", "contradiction"]
    conf = {g: {s: 0 for s in labels + ["none"]} for g in labels}
    for ex in examples:
        conf[ex["label"]][stance(ex["explanation"])] += 1
    n = sum(sum(r.values()) for r in conf.values())
    acc = sum(conf[l][l] for l in labels) / max(n, 1)
    return acc, conf
