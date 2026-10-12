# Released MDLM backbone

`backbone.py` exposes the checkpoint used by the additive-chain experiments,
not the older checkpoint used by this repository's original `main.py`.

- Upstream: `kuleshov-group/bd3lms`, commit
  `1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7`.
- Hugging Face model: `kuleshov-group/mdlm-owt-noeos`, revision
  `b9613c3c54da477d198c998e767ac3647e01225b`.
- Tokenizer: `gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`,
  with existing local tokenizer files.
- Measured runtime: Python 3.10, PyTorch 2.5.1+cu121, Lightning 2.5.0.post0,
  Transformers 4.49.0, Triton 3.1.0. These differ from the old root environment.

Prepare those assets on the compute machine. The loader takes local paths,
checks the checkpoint, five tokenizer files and six core upstream source files
against recorded hashes, and performs no explicit download or checkpoint write. Use a clean
upstream checkout: the six-file check does not authenticate its entire tree.
The checkpoint includes Python model code; the verified upstream loader uses
`trust_remote_code=True` to execute that **local pinned code**.

The required tokenizer files are `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json` and `merges.txt`. Root-level
`special_tokens_map.json` and `added_tokens.json` are rejected: they are not in
this original snapshot and can override the authenticated token semantics.
Unrelated weights, documentation and subdirectories may remain in the snapshot.
The loader records the tokenizer revision, file hashes and observed token IDs.

Upstream creates `[PAD]` in memory at ID 50,257, so `len(tokenizer)` becomes
50,258 while its native `vocab_size` stays 50,257. BOS, EOS and UNK are 50,256;
the tokenizer has no MASK token. The diffusion model uses 50,257 as its implicit
mask ID and has vocabulary size 50,258. These values are checked explicitly.
Do not save the augmented tokenizer back into the authenticated source snapshot.

Start a fresh Python process to avoid mixing the two repositories' unqualified
`diffusion`, `models`, and related imports:

```python
import torch
from additive_chain.backbone import load_bd3lm

frozen = load_bd3lm(
    upstream="/path/to/bd3lms",
    checkpoint="/path/to/mdlm-owt-noeos-snapshot",
    tokenizer="/path/to/gpt2-tokenizer-snapshot",
    device="cuda", steps=32, top_p=0.9,
)
tokens = torch.full((1, 1024), frozen.mask_id, device="cuda", dtype=torch.long)
tokens[:, 0] = frozen.model.tokenizer.bos_token_id
output = frozen.forward(tokens, 0.5)
# output.log_probs: native [batch, length, vocabulary] FP32 values.
# output.hidden: detached features before final output norm/projection.
```

EMA is copied before freezing parameters. A forward hook observes one native
backbone call and is removed on success or failure. Gold tokens are never
inputs to this feature interface. Detached features can feed a trainable head
without retaining a backbone graph.

The config keeps article EOS and disables artificial window BOS/EOS for data
packing. This does not itself build a token dataset. Generation still uses the
upstream initial boundary token; it is not an extra training-window wrapper.

This adapter alone is **not** a full generation/training command. Candidate
construction, chain inference, reveal scheduling, rejection accounting and
scoring are separate. CPU tests verify the one-forward/hook contract with a
small native model; they do not establish full-checkpoint GPU parity or a
generation-speed result for this public adapter.
