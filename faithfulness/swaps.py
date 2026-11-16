"""Word substitutions for the evidence-contradiction intervention.

`Swapper.substitutes(token)` returns words that describe something *different and
usually incompatible* in the same slot: man -> woman, red -> blue, sitting -> standing,
dog -> cat. Each substitute is inflected like the token, keeps its capitalisation and
punctuation, and is a word that occurs in e-SNLI, so the edited sentence stays natural.

Two sources:
  1. CURATED groups (gender, age, colour, number, posture, place/time, common adjectives).
     Every member of a group is incompatible with the others.
  2. WordNet nouns: co-hyponyms (siblings under the same parent), e.g. pool -> lake,
     guitar -> piano. A sibling is used only if that sense is the candidate word's most
     frequent sense (so "head" is never offered as a domestic animal) and it is not
     a hypernym or hyponym of the original word (dog -> puppy is not a contradiction).

Substitutes are *candidates*. Whether an edit really changes the label is decided
later by an independent NLI model (the oracle, see predictors.NLIOracle) and by
manual spot-checks, not by this module.

`Swapper.related(a, b)` says whether b is one of a's substitutes (pool/ocean,
man/woman). The "align" edit uses it to find the clashing word pair in a
contradiction (premise "pool" vs hypothesis "ocean").
"""
import re
from collections import Counter

CURATED = [
    # gender (age-matched) and roles
    ["man", "woman"], ["men", "women"], ["boy", "girl"], ["boys", "girls"], ["male", "female"],
    ["males", "females"], ["husband", "wife"], ["father", "mother"], ["son", "daughter"],
    ["brother", "sister"], ["gentleman", "lady"], ["gentlemen", "ladies"], ["king", "queen"],
    ["he", "she"], ["his", "her"],
    # age
    ["child", "adult"], ["children", "adults"], ["kid", "adult"], ["kids", "adults"], ["young", "old", "elderly"],
    ["baby", "teenager"], ["babies", "teenagers"],
    # common animals (WordNet puts dog and cat under different parents)
    ["dog", "cat", "horse", "cow", "sheep", "bird"], ["dogs", "cats", "horses", "cows", "birds"],
    # colours
    ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray"],
    # numbers
    ["two", "three", "four", "five", "six", "seven"],
    # posture / activity
    ["sitting", "standing", "lying", "running", "sleeping"], ["sits", "stands", "runs", "sleeps"],
    ["sit", "stand", "run", "sleep"], ["walking", "sitting"], ["walks", "sits"],
    ["asleep", "awake"], ["laughing", "crying"], ["smiling", "frowning"],
    # place / time / weather
    ["indoors", "outdoors"], ["inside", "outside"], ["indoor", "outdoor"], ["day", "night"],
    ["daytime", "nighttime"], ["morning", "evening"], ["summer", "winter"], ["sunny", "rainy", "snowy"],
    ["beach", "mountain"], ["city", "countryside"], ["urban", "rural"],
    # common adjectives
    ["empty", "crowded"], ["wet", "dry"], ["hot", "cold"], ["happy", "sad"], ["open", "closed"],
    ["tall", "short"], ["big", "small"], ["large", "small"], ["fast", "slow"], ["clean", "dirty"],
    ["full", "empty"], ["light", "dark"], ["loud", "quiet"], ["alone", "together"], ["new", "old"],
]
_ALIASES = {"grey": "gray"}
_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$")
_STOP_NOUNS = {"thing", "something", "one", "other", "people", "person", "group", "lot", "front", "top",
               "side", "area", "way", "part", "kind", "type", "background", "picture", "image"}
_IRREGULAR_PLURALS = {"men", "women", "children", "people", "feet", "teeth", "mice", "geese", "gentlemen"}
# WordNet noun categories where a sibling is a different, concrete thing. People are left to the
# curated groups (WordNet person nouns are mostly "kind of" relations: son/man, worker/man), and
# acts only count when they are sports or games (soccer/baseball, not market/protection).
_CONCRETE = {"noun.artifact", "noun.animal", "noun.food", "noun.object", "noun.plant", "noun.location",
             "noun.body", "noun.substance", "noun.act"}
_ACT_ROOTS = {"sport.n.01", "game.n.01", "dance.n.01"}


def split_token(token):
    """('"', 'Dogs', '.') for '"Dogs.'"""
    m = _SPLIT.match(token)
    return m.group(1), m.group(2), m.group(3)


def _pluralise(w):
    if w.endswith(("s", "sh", "ch", "x", "z")):
        return w + "es"
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _match_case(src, w):
    if src.isupper() and len(src) > 1:
        return w.upper()
    if src[:1].isupper():
        return w[:1].upper() + w[1:]
    return w


class Swapper:
    def __init__(self, vocab_sentences=(), min_count=3, use_wordnet=True):
        """vocab_sentences: sentences used to decide which words are common enough to insert
        (we pass all premises and hypotheses of the split)."""
        self.vocab = Counter(split_token(t)[1].lower() for s in vocab_sentences for t in s.split())
        self.min_count = min_count
        self.groups = {}
        for g in CURATED:
            for w in g:
                self.groups.setdefault(w, set()).update(x for x in g if x != w)
        self.wn = None
        if use_wordnet:
            try:
                from nltk.corpus import wordnet as wn
                wn.synsets("dog")  # raises LookupError if the corpus is missing
                self.wn = wn
            except (ImportError, LookupError) as e:
                print(f"[swaps] WordNet unavailable ({type(e).__name__}); using curated substitutions only. "
                      "Fix: pip install nltk && python -m nltk.downloader wordnet")
        self._cache = {}
        self._rel_cache = {}
        self._sense_cache = {}

    def _common(self, w):
        return not self.vocab or self.vocab[w] >= self.min_count

    # ------------------------------------------------------------ WordNet nouns
    def _noun_senses(self, word, k=1):
        """(lemma, top-k concrete noun senses) if `word` is mainly used as a common noun, else (None, [])."""
        if (word, k) not in self._sense_cache:
            self._sense_cache[(word, k)] = self._noun_senses_uncached(word, k)
        return self._sense_cache[(word, k)]

    def _noun_senses_uncached(self, word, k):
        wn = self.wn
        if wn is None or word in _STOP_NOUNS:
            return None, []
        lemma = wn.morphy(word, wn.NOUN)
        if not lemma or lemma in _STOP_NOUNS:
            return None, []
        senses = wn.synsets(lemma, wn.NOUN)
        if not senses or any(l.name()[:1].isupper() for l in senses[0].lemmas() if l.name().lower() == lemma):
            return None, []  # proper noun / nationality ("Japanese", "American")

        def freq(pos):
            base = wn.morphy(word, pos)
            if not base:
                return 0, 0
            ss = wn.synsets(base, pos)
            return sum(l.count() for s in ss for l in s.lemmas() if l.name().lower() == base), len(ss)

        n_cnt, n_syn = freq(wn.NOUN)
        o = [freq(p) for p in (wn.VERB, wn.ADJ, wn.ADV)]
        o_cnt, o_syn = sum(c for c, _ in o), sum(s for _, s in o)
        if (n_cnt or o_cnt) and o_cnt > n_cnt:  # "get", "swimming", "present", "working": mainly not nouns
            return None, []
        if not (n_cnt or o_cnt) and o_syn > n_syn:
            return None, []
        if not self._is_concrete(senses[0]):
            return None, []  # the word's main meaning must be a concrete thing ("guy" is a person)
        return lemma, [s for s in senses[:k] if self._is_concrete(s)]

    @staticmethod
    def _is_concrete(sense):
        if sense.lexname() not in _CONCRETE:
            return False
        if sense.lexname() == "noun.act":
            return bool({s.name() for s in sense.closure(lambda s: s.hypernyms())} & _ACT_ROOTS)
        return True

    def _noun_alternatives(self, word):
        """Siblings of the main noun sense, as lower-case surface words inflected like `word`."""
        wn = self.wn
        lemma, senses = self._noun_senses(word, k=1)
        if not senses:
            return set()
        s0 = senses[0]
        related = set(s0.closure(lambda s: s.hypernyms())) | set(s0.closure(lambda s: s.hyponyms(), depth=3))
        plural = lemma != word
        out = set()
        for parent in s0.hypernyms():
            for sib in parent.hyponyms():
                if sib == s0 or sib in related:
                    continue
                for l in sib.lemmas():
                    c = l.name()
                    if "_" in c or "-" in c or not c.isalpha() or not c.islower() or len(c) < 3 \
                            or c == lemma or c in _STOP_NOUNS:
                        continue
                    if self._noun_senses(c)[1][:1] != [sib]:  # must be the candidate's main, noun meaning
                        continue
                    out.add(_pluralise(c) if plural else c)
        return out

    def _noun_clash(self, a, b):
        """WordNet: a and b are different things of the same kind, up to cousins (pool / ocean)."""
        _, sa = self._noun_senses(a, k=3)
        _, sb = self._noun_senses(b, k=3)
        for x in sa:
            up_x = set(x.closure(lambda s: s.hypernyms()))
            for y in sb:
                if x == y or y in up_x or x in set(y.closure(lambda s: s.hypernyms())):
                    continue  # same thing, or one is a kind of the other (dog / animal)
                d = x.shortest_path_distance(y)
                lch = x.lowest_common_hypernyms(y)
                # depth >= 3 admits e.g. body_of_water (pond / ocean) but not object / whole
                if d is not None and d <= 3 and lch and lch[0].min_depth() >= 3:
                    return True
        return False

    def alternatives(self, word):
        """All incompatible alternatives for a bare lower-case word (no frequency filter)."""
        word = _ALIASES.get(word, word)
        if word not in self._cache:
            alts = set(self.groups.get(word, ()))
            if not alts:
                alts = self._noun_alternatives(word)
            self._cache[word] = alts
        return self._cache[word]

    # ------------------------------------------------------------ public API
    def substitutes(self, token):
        """Surface substitutes for `token` (punctuation and capitalisation preserved), sorted."""
        pre, core, post = split_token(token)
        if not core.isalpha():
            return []
        alts = sorted(a for a in self.alternatives(core.lower()) if self._common(a))
        return [pre + _match_case(core, a) + post for a in alts]

    def related(self, a, b):
        """True if word b is an incompatible alternative of word a (e.g. pool/ocean, man/woman)."""
        a, b = split_token(a)[1].lower(), split_token(b)[1].lower()
        if not a.isalpha() or not b.isalpha() or a == b:
            return False
        key = (a, b) if a < b else (b, a)
        if key not in self._rel_cache:
            self._rel_cache[key] = (b in self.alternatives(a) or a in self.alternatives(b)
                                    or self._noun_clash(a, b))
        return self._rel_cache[key]

    def _is_plural(self, w):
        if w in _IRREGULAR_PLURALS:
            return True
        if self.wn is None:
            return w.endswith("s") and not w.endswith("ss")
        lemma = self.wn.morphy(w, self.wn.NOUN)
        return bool(lemma) and lemma != w

    def transplant(self, token, donor):
        """`donor`'s word written in `token`'s place: same number (singular/plural) as `token`,
        token's capitalisation and punctuation. "ship" + "cruisers" -> "cruiser"."""
        pre, core, post = split_token(token)
        w = split_token(donor)[1].lower()
        if self.wn is not None and w not in self.groups:
            tp, dp = self._is_plural(core.lower()), self._is_plural(w)
            if dp and not tp:
                w = self.wn.morphy(w, self.wn.NOUN)
            elif tp and not dp:
                w = _pluralise(w)
        return pre + _match_case(core, w) + post
