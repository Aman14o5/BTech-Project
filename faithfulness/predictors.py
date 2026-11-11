"""Self-rationalizing predictors: (premise, hypothesis) -> label, label probs, explanation.

T5Predictor reproduces the input/output format and label scoring of the
original repo (code/feature_conversion_methods.py and code/Inference.py):
  input : "explain nli hypothesis: {h} premise: {p}"
  output: "{label} "explanation: " {explanation}"
The label is the argmax over the first-generated-token probabilities of the
three label words (as in Inference.py); we renormalise those three
probabilities so they sum to 1, which makes confidence shifts comparable.
"""
import re

LABELS = ["entailment", "neutral", "contradiction"]


def _split_label_explanation(text):
    parts = re.split(r'"?\s*explanation:\s*"?', text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(' "'), parts[1].strip(' "')
    words = text.split()
    return (words[0] if words else ""), " ".join(words[1:])


class T5Predictor:
    def __init__(self, model_path, batch_size=32, max_new_tokens=100, device=None):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        except (OSError, ValueError, TypeError):
            # checkpoint saved without tokenizer files: fine-tuning does not change the T5 vocab
            print("No tokenizer in checkpoint; using the base t5-large tokenizer")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        # first sub-token of each label word; for T5 this gives [3, 7163, 27252],
        # the same ids hard-coded in code/Inference.py
        self.label_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
        assert len(set(self.label_ids)) == 3, f"label first-token ids collide: {self.label_ids}"

    def predict(self, pairs):
        torch = self.torch
        out = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            inputs = [f"explain nli hypothesis: {h} premise: {p}" for p, h in batch]
            enc = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          output_scores=True, return_dict_in_generate=True)
            first = torch.softmax(gen.scores[0].float(), dim=-1)[:, self.label_ids]
            first = (first / first.sum(dim=-1, keepdim=True)).cpu().tolist()
            texts = self.tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
            for probs, text in zip(first, texts):
                gen_label, expl = _split_label_explanation(" ".join(text.split()))
                out.append({
                    "label": LABELS[max(range(3), key=lambda i: probs[i])],
                    "probs": dict(zip(LABELS, probs)),
                    "generated_label": gen_label,
                    "explanation": expl,
                })
        return out


class DummyPredictor:
    """Rule-based stand-in used only to test the pipeline without a GPU/model.
    Its numbers mean nothing; never report them."""

    def predict(self, pairs):
        from evidence import content_indices, normalise
        out = []
        for p, h in pairs:
            pw = {normalise(w) for w in p.split()}
            hw = [h.split()[i] for i in content_indices(h)]
            shared = [w for w in hw if normalise(w) in pw]
            ratio = len(shared) / max(len(hw), 1)
            label = "entailment" if ratio > 0.6 else "neutral" if ratio > 0.3 else "contradiction"
            probs = {l: 0.1 for l in LABELS}
            probs[label] = 0.8
            out.append({"label": label, "probs": probs, "generated_label": label,
                        "explanation": " ".join(shared) or "no overlap"})
        return out
