"""Self-rationalizing predictors: (premise, hypothesis) -> label, label probs, explanation.

T5Predictor reproduces the input/output format and label scoring of the
original repo (code/feature_conversion_methods.py and code/Inference.py):
  input : "explain nli hypothesis: {h} premise: {p}"
  output: "{label} "explanation: " {explanation}"
The label is the argmax over the first-decoder-step probabilities of the three
label words (as in Inference.py); we renormalise those three probabilities so
they sum to 1, which makes confidence shifts comparable.

Two entry points:
  predict(pairs)        full generation: label, probs AND explanation (slow)
  predict_labels(pairs) one decoder step: label and probs only (~20-50x faster);
                        gives exactly the same label/probs as predict(), because
                        generate()'s first-step scores come from the same forward pass.
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
    def __init__(self, model_path, batch_size=32, label_batch_size=128, max_new_tokens=100, device=None):
        import os
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        if not os.path.isfile(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(
                f"No checkpoint at '{model_path}': expected a folder containing config.json and "
                "pytorch_model.bin / model.safetensors. Check MODEL_PATH (e.g. run: !ls \"$MODEL_PATH\").")
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
        self.label_batch_size = label_batch_size
        self.max_new_tokens = max_new_tokens
        # first sub-token of each label word; for T5 this gives [3, 7163, 27252],
        # the same ids hard-coded in code/Inference.py
        self.label_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
        assert len(set(self.label_ids)) == 3, f"label first-token ids collide: {self.label_ids}"

    def _encode(self, batch):
        inputs = [f"explain nli hypothesis: {h} premise: {p}" for p, h in batch]
        return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                              max_length=512).to(self.device)

    def _probs(self, first_step_logits):
        p = self.torch.softmax(first_step_logits.float(), dim=-1)[:, self.label_ids]
        return (p / p.sum(dim=-1, keepdim=True)).cpu().tolist()

    @staticmethod
    def _pack(probs, gen_label="", expl=None):
        out = {"label": LABELS[max(range(3), key=lambda i: probs[i])], "probs": dict(zip(LABELS, probs))}
        if expl is not None:
            out.update(generated_label=gen_label, explanation=expl)
        return out

    def _batched(self, pairs, bs, fn):
        """Run fn over length-sorted batches (less padding) and restore input order."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [None] * len(pairs)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            for i, r in zip(idx, fn([pairs[i] for i in idx])):
                out[i] = r
        return out

    def predict(self, pairs):
        def run(batch):
            enc = self._encode(batch)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          output_scores=True, return_dict_in_generate=True)
            texts = self.tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
            res = []
            for probs, text in zip(self._probs(gen.scores[0]), texts):
                gl, ex = _split_label_explanation(" ".join(text.split()))
                res.append(self._pack(probs, gl, ex))
            return res
        return self._batched(pairs, self.batch_size, run)

    def predict_labels(self, pairs):
        torch = self.torch

        def run(batch):
            enc = self._encode(batch)
            start = torch.full((len(batch), 1), self.model.config.decoder_start_token_id,
                               dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(**enc, decoder_input_ids=start).logits[:, 0, :]
            return [self._pack(p) for p in self._probs(logits)]
        return self._batched(pairs, self.label_batch_size, run)


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
            expl = {"entailment": "the {} is {}", "neutral": "just because {} doesn't mean {}",
                    "contradiction": "the {} cannot be {}"}[label].format(" ".join(shared) or "x", h.split()[-1])
            out.append({"label": label, "probs": probs, "generated_label": label, "explanation": expl})
        return out

    def predict_labels(self, pairs):
        return [{"label": r["label"], "probs": r["probs"]} for r in self.predict(pairs)]
