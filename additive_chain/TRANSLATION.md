# Conditional translation pilot

The IWSLT experiment adapts the pinned MDLM checkpoint to German-to-English
translation, then freezes it for independent and chain-CRF decoding. It does
not reuse an unconditional generator as an already trained translator.

`iwslt_data` prepares the fairseq IWSLT14 split using a supplied Moses checkout
and the existing GPT-2 tokenizer. It records the archive, preprocessing and
tokenizer hashes, bilingual row IDs, split sizes, exact duplicate counts, and
training-target bigram counts. No edge connects different sentences. The
standard splits are preserved, including their recorded duplicate pairs.
All prepared examples must fit the 1,024-token context; preparation fails
rather than silently removing or truncating a long example.

The input is `EOS German EOS English EOS`. German is always visible, and
only English tokens and its final EOS receive masking and denoising loss.
Padding is excluded from attention keys and loss. The conditional forward
uses the released DDiT layers, with an explicit padding mask. Tests compare
it to the original unpadded forward and to batches with different padding.
Training samples mask probability uniformly from [.001,.999] and weights
masked-token cross entropy by its reciprocal. The denominator includes all
target tokens in the effective batch, including targets that were not masked.

At each development checkpoint, a small length classifier is fitted on
source-only features from training pairs, with the backbone frozen. This
allows checkpoint selection by translation BLEU without supplying reference
lengths. Generation receives only the source, the length classifier and the
frozen model. It cannot read target tokens. The length classifier is saved
with the selected backbone.

## Local-assets commands

Paths below are placeholders for user-supplied local directories. These
commands do not download model weights or submit cluster jobs.

```sh
python -m additive_chain.iwslt_data \
  --archive /data/de-en.tgz --moses /data/mosesdecoder \
  --tokenizer /data/gpt2 --output /data/iwslt

python -m additive_chain.iwslt_run gate \
  --checkpoint /models/mdlm-owt-noeos --tokenizer /data/gpt2 \
  --output /results/translation-gate

python -m additive_chain.iwslt_run train \
  --checkpoint /models/mdlm-owt-noeos --tokenizer /data/gpt2 \
  --data /data/iwslt --output /results/pilot-lr1e4 \
  --lr 0.0001 --updates 10000 --seconds-limit 36000

python -m additive_chain.iwslt_compare \
  --checkpoint /models/mdlm-owt-noeos --tokenizer /data/gpt2 \
  --data /data/iwslt --trained /results/pilot-lr1e4/best.pt \
  --output /results/comparison --limit 128
```

The runner keeps a resumable optimizer/RNG checkpoint and the best development
checkpoint. A run-directory lock prevents concurrent writers. It refuses to
resume with changed training source, settings or data. The tiny-data smoke
mode must reduce full-mask loss by at least 35%; this is an implementation
test, not a translation-quality result.

The development comparison holds the backbone, length predictor, candidate
sets, CRF scores and reveal random streams fixed between joint sampling and
independent draws from exact CRF marginals. Temperature applies to the entire
chain energy, after selecting candidates from the original model scores.
Temperature one preserves the existing decoder path. Candidate selection
uses top-p .9 intersected with top-K 500. Source positions never receive CRF
edges. Intermediate unrevealed target positions remain part of the chain.

Outputs include every generated translation, corpus SacreBLEU and chrF,
actual forward counts, predicted lengths and elapsed time. Pilot timings
include first-call warmup and are explicitly not final speed measurements.
`iwslt_readout` compares corpus BLEU and chrF with paired sentence-bootstrap
intervals. It rejects mixed checkpoint identities or mismatched sentences.
These development-set intervals describe the pilot, not a final test claim.

`iwslt_spans` measures the separate fixed-input comparison: choose development
sentences and a 2-, 4-, or 8-token target span before model scoring, then mask
that span while leaving the German source and other English words visible.
For identical candidate sets and CRF scores, it compares the gold span's
joint probability with the product of its exact token marginals, and saves
draws from both distributions. Immediate visible target neighbors are clamped;
there is no source-to-target CRF edge. Excluded gold tokens have probability
zero under both methods and are not removed from the population. This is a
reconstruction diagnostic with reference context, not free translation.

COMET, controlled batch-size timing and learned-head comparisons are subsequent
evaluations, not results established by running this pilot.
