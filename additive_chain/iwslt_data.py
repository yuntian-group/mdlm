"""Prepare the fairseq IWSLT14 De-En split with the pinned MDLM tokenizer.

Moses is an explicit local checkout. Data, source hashes, IDs and counts are
saved outside the code repository. No dataset is uploaded by this command.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

import torch

from .backbone import file_sha256, verify_tokenizer_files
from .translation import CONTEXT_LENGTH, EOS_ID, MASK_ID


def unix_lines(text):
    # The source corpus contains U+0085 inside a German sentence. splitlines()
    # would treat it as a row boundary and break the bilingual alignment.
    return text.removesuffix("\n").split("\n")


def filter_training(lines):
    result = []
    for line in lines:
        if any(tag in line for tag in ("<url>", "<talkid>", "<keywords>")):
            continue
        result.append(re.sub(r"</?(title|description)>", "", line).strip())
    return result


def xml_segments(text):
    return [m.replace("’", "'") for m in re.findall(
        r'<seg id="[0-9]+">\s*(.*?)\s*</seg>', text)]


def prepare(args):
    from transformers import GPT2TokenizerFast
    verify_tokenizer_files(args.tokenizer)
    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    if args.output.exists():
        raise ValueError("Output already exists; do not replace a frozen dataset")
    scripts = args.moses / "scripts"
    used_scripts = [scripts / "tokenizer/tokenizer.perl", scripts / "tokenizer/lowercase.perl",
                    scripts / "training/clean-corpus-n.perl"]
    def perl(path, lines, *options):
        proc = subprocess.run(["perl", str(path), *options],
            input="\n".join(lines) + "\n", text=True, capture_output=True, check=True)
        return unix_lines(proc.stdout)
    def tokenize(lines, lang):
        return perl(used_scripts[0], lines, "-threads", "2", "-l", lang)
    def lowercase(lines):
        return perl(used_scripts[1], lines)
    with tarfile.open(args.archive) as archive:
        names = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
        def read(name):
            return archive.extractfile(names[name]).read().decode("utf-8")
        train = {lang: tokenize(filter_training(unix_lines(read(f"train.tags.de-en.{lang}"))), lang)
                 for lang in ("de", "en")}
        if len(train["de"]) != len(train["en"]):
            raise ValueError("Raw training alignment mismatch")
        with tempfile.TemporaryDirectory(prefix="iwslt-clean-") as tmp:
            raw, clean = Path(tmp) / "raw", Path(tmp) / "clean"
            for lang in ("de", "en"):
                Path(f"{raw}.{lang}").write_text("\n".join(train[lang]) + "\n")
            subprocess.run(["perl", str(used_scripts[2]), "-ratio", "1.5", str(raw),
                            "de", "en", str(clean), "1", "175"], check=True)
            train = {lang: lowercase(unix_lines(Path(f"{clean}.{lang}").read_text()))
                     for lang in ("de", "en")}
        rows = {"train": [], "dev": [], "test": []}
        for i, (source, target) in enumerate(zip(train["de"], train["en"]), 1):
            split = "dev" if i % 23 == 0 else "train"
            rows[split].append(dict(id=f"train-clean-{i}", source=source, target=target))
        for name in ("IWSLT14.TED.dev2010", "IWSLT14.TEDX.dev2012", "IWSLT14.TED.tst2010",
                     "IWSLT14.TED.tst2011", "IWSLT14.TED.tst2012"):
            pair = {lang: lowercase(tokenize(xml_segments(read(f"{name}.de-en.{lang}.xml")), lang))
                    for lang in ("de", "en")}
            if len(pair["de"]) != len(pair["en"]):
                raise ValueError("Test alignment mismatch")
            for i, (source, target) in enumerate(zip(pair["de"], pair["en"]), 1):
                rows["test"].append(dict(id=f"{name}-{i}", source=source, target=target))
    args.output.mkdir(parents=True)
    manifest = dict(schema="iwslt14_mdlm_v1", archive_sha256=file_sha256(args.archive),
        moses_commit=subprocess.check_output(["git", "-C", str(args.moses), "rev-parse", "HEAD"], text=True).strip(),
        preprocessing_hashes={str(p.relative_to(args.moses)): file_sha256(p) for p in used_scripts},
        tokenizer_hashes=verify_tokenizer_files(args.tokenizer),
        format="EOS source EOS target EOS; source visible, target-only noise/loss",
        context_length=CONTEXT_LENGTH, split_rule="fairseq prepare-iwslt14.sh; no BPE learning",
        long_input_policy="Reject preparation if any pair exceeds 1024; no silent truncation or dropping",
        splits={}, duplicate_pairs={})
    pair_sets = {}
    counts = Counter()
    for split, records in rows.items():
        src_ids = tok([r["source"] for r in records], add_special_tokens=False)["input_ids"]
        tgt_ids = tok([r["target"] for r in records], add_special_tokens=False)["input_ids"]
        for row, src, tgt in zip(records, src_ids, tgt_ids):
            if not src or not tgt or len(src) + len(tgt) + 3 > CONTEXT_LENGTH:
                raise ValueError(f"Explicit length policy required for {row['id']}: {len(src)}, {len(tgt)}")
            if any(x >= EOS_ID for x in src + tgt):
                raise ValueError("Text contains reserved EOS/MASK")
            row.update(source_ids=src, target_ids=tgt)
            if split == "train":
                tokens = tgt + [EOS_ID]
                counts.update(a * (MASK_ID + 1) + b for a, b in zip(tokens, tokens[1:]))
        path = args.output / f"{split}.jsonl"
        with path.open("x") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pair_sets[split] = {(r["source"], r["target"]) for r in records}
        manifest["splits"][split] = dict(rows=len(records), sha256=file_sha256(path),
            max_source_tokens=max(map(len, src_ids)), max_target_tokens=max(map(len, tgt_ids)),
            unique_pairs=len(pair_sets[split]))
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        manifest["duplicate_pairs"][f"{a}:{b}"] = len(pair_sets[a] & pair_sets[b])
    keys = torch.tensor(sorted(counts), dtype=torch.long)
    values = torch.tensor([counts[k] for k in keys.tolist()], dtype=torch.float64)
    left = torch.zeros(MASK_ID + 1, dtype=torch.float64)
    right = torch.zeros_like(left)
    left.index_add_(0, keys // (MASK_ID + 1), values)
    right.index_add_(0, keys % (MASK_ID + 1), values)
    torch.save(dict(left_counts=left, right_counts=right, pair_keys=keys, pair_counts=values),
               args.output / "counts.pt")
    manifest["counts_sha256"] = file_sha256(args.output / "counts.pt")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "moses", "tokenizer", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
