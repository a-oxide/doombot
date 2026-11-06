"""prompt, letter readout, and model loading shared by the entry points."""

import torch

# fixed by the weights: the adapter was trained against this exact question, and
# changing the wording moves the argmax on most decisions
QUESTION = "Which category best describes this customer request?"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def prompt(state, options, letters):
    lines = ["State: " + state, "Question: " + QUESTION]
    lines += ["%s) %s" % (ch, text) for ch, text in zip(letters, options)]
    return "\n".join(lines + ["Answer:"])


def letter_ids(tok, letters):
    ids, used = [], []
    for ch in letters:
        for cand in (ch, " " + ch):
            t = tok.encode(cand, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                used.append(cand)
                break
        else:
            raise SystemExit("no single token for option letter %r" % ch)
    return ids, used


def load(name, adapter=None):
    import transformers
    from transformers import AutoTokenizer

    # qwen ships a processor; older checkpoints only have a tokenizer
    try:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(name)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"

    cls = getattr(transformers, "AutoModelForCausalLM", None) or getattr(transformers, "AutoModel")
    model = cls.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def last_logits(model, enc):
    """logits at the final position only; transformers 4.45+ can skip the rest."""
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        return model(**enc).logits[:, -1, :].float()


@torch.no_grad()
def probs(model, tok, ids, text):
    enc = tok([text], return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to("cuda") for k, v in enc.items()}
    return torch.softmax(last_logits(model, enc)[:, ids], dim=-1)[0]


def choose(model, tok, ids, text):
    p = probs(model, tok, ids, text)
    a = int(p.argmax())
    return a, float(p[a])
