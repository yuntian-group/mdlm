"""Paired context interventions for a frozen two-step translation model.

One global-joint trajectory supplies the states for both heads. Target length
is the reference length, including EOS. At the second step, a paired state
replaces only visible target identities by their aligned reference tokens.
This measures reconstruction sensitivity, not whether a generated paraphrase
is incorrect. The initial all-masked state is scored separately.

Each state uses its actual backbone top-p/K support, shared by both heads.
There is no gold insertion or union across states. Unsupported gold has
infinite original-support NLL (JSON null plus an explicit status). Conditional
finite summaries always report their covered population and token denominator.

Absolute imports allow this runner to use a previously frozen evaluator on
PYTHONPATH. Its source is authenticated against an original selection receipt.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from additive_chain import chain
from additive_chain import iwslt_residual_eval as evaluation
from additive_chain.adaptation import count_pair_scores
from additive_chain.backbone import file_sha256
from additive_chain.composition import compose_pair_scores
from additive_chain.fast_candidates import extract_compact_candidates
from additive_chain.generation import sample_window, stream_seed
from additive_chain.iwslt_confirm import atomic_json, read
from additive_chain.iwslt_followup import output_name, stratified_rows, verify_output
from additive_chain.translation import CONTEXT_LENGTH, EOS_ID, MASK_ID


FORMAT = "iwslt_context_replay_v1"
EXAMPLES = 512
FULL_EXAMPLES = 7283
SELECTION_EXAMPLES = 1024
PROTOCOL = dict(steps=2, seed=101, nucleus_p=.9, top_k=500, temperature=.85,
                count_strength=.2, residual_strength=1., length="reference_including_eos",
                rollout="selected_global_joint", support="original_per_state_no_gold_insertion",
                stream_index="original_full_development_position")
STATE_NAMES = ("first_all_masked", "later_generated_visible", "later_gold_visible")
MODEL_NAMES = ("global", "contextual")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def selected_heads(context_document, global_document, common):
    """Validate existing decisions without creating a new checkpoint selection."""
    official = evaluation.validate_selected(context_document, common, "contextual")
    global_common = dict(common, inherited_global_sha256=None)
    parent = evaluation.validate_selected(global_document, global_common, "global")
    for document in (context_document, global_document):
        if (document.get("examples") != SELECTION_EXAMPLES
                or document["manifest"].get("complete") is not True):
            raise ValueError("Replay requires complete development checkpoint selections")
    if (official["name"], official["epoch"], official["update"]) != ("epoch-0000", 0, 0):
        raise ValueError("This diagnostic requires the official zero/global fallback")
    contextual = next((row["candidate"] for row in context_document["ranking"]
                       if row["candidate"]["update"] > 0), None)
    if (contextual is None or contextual["name"] != "epoch-0001"
            or contextual["epoch"] != 1):
        raise ValueError("The requested trained diagnostic is the ranked epoch-one head")
    if parent["update"] <= 0 or parent["sha256"] != common["inherited_global_sha256"]:
        raise ValueError("Contextual parent is not the selected learned global head")
    return official, contextual, parent


def prepare(args):
    common = evaluation.common_assets(args)
    context_document, global_document = read(args.contextual_selection), read(args.global_selection)
    official, candidate, parent = selected_heads(context_document, global_document, common)
    if Path(parent["head"]).resolve() != args.global_head.resolve():
        raise ValueError("Pass the original selected global checkpoint path")
    # Authenticate the imported frozen package, not just the runner's filename.
    cell = evaluation.cells("selection")[0]
    relative = f"{official['name']}/selection/{output_name(cell)}/result.json"
    reference = args.official_evaluation_root / relative
    if file_sha256(reference) != context_document["outputs"].get(relative):
        raise ValueError("Original evaluator reference changed")
    selection_rows = evaluation.development_rows(args.data, "selection")
    if len(selection_rows) != SELECTION_EXAMPLES:
        raise ValueError("Original selection population changed")
    verify_output(reference, evaluation.build_identity(
        cell, selection_rows, "selection", "contextual", official, common))
    head, inherited, metadata = evaluation.load_heads(
        Path(candidate["head"]), "contextual", common, args.global_head)
    zero, _, zero_metadata = evaluation.load_heads(
        Path(official["head"]), "contextual", common, args.global_head)
    if (inherited is None or evaluation.candidate_entry(candidate["head"], metadata) != candidate
            or evaluation.candidate_entry(official["head"], zero_metadata) != official):
        raise ValueError("Contextual checkpoint identity changed")
    _, _, parent_metadata = evaluation.load_heads(
        args.global_head, "global", dict(common, inherited_global_sha256=None))
    if evaluation.candidate_entry(args.global_head, parent_metadata) != parent:
        raise ValueError("Global checkpoint identity changed")
    full = evaluation.development_rows(args.data, "full")
    if len(full) != FULL_EXAMPLES or len({row["id"] for row in full}) != FULL_EXAMPLES:
        raise ValueError("Replay requires the original unique full-development population")
    rows = stratified_rows(full, EXAMPLES)
    positions = {row["id"]: index for index, row in enumerate(full)}
    if len(rows) != EXAMPLES or len({row["id"] for row in rows}) != EXAMPLES:
        raise ValueError("Invalid fixed replay population")
    paths = dict(contextual_selection=args.contextual_selection, global_selection=args.global_selection,
                 reference_output=reference, reference_identity=reference.with_name("result-identity.json"))
    identity = dict(format=FORMAT, role="diagnostic", split="dev", protocol=PROTOCOL,
        examples=EXAMPLES, ids=[row["id"] for row in rows],
        stream_indices=[positions[row["id"]] for row in rows],
        population_sha256=digest(rows), common=common, official_contextual=official,
        contextual_diagnostic=candidate, selected_global=parent,
        input_sha256={key: file_sha256(path) for key, path in paths.items()},
        source_sha256=evaluation.source_hashes(), runner_sha256=file_sha256(Path(__file__)),
        contextual_training_source_sha256=metadata["identity"]["source_sha256"],
        global_training_source_sha256=parent_metadata["identity"]["source_sha256"],
        frozen_reference_relative_path=relative)
    for path in paths.values():
        if path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Inputs must remain outside replay outputs")
    return SimpleNamespace(common=common, rows=rows, positions=positions, head=head,
                           inherited=inherited, zero=zero, identity=identity, paths=paths)


def unchanged(args, state):
    identity = state.identity
    if (evaluation.common_assets(args) != state.common
            or {key: file_sha256(path) for key, path in state.paths.items()} != identity["input_sha256"]
            or evaluation.source_hashes() != identity["source_sha256"]
            or file_sha256(Path(__file__)) != identity["runner_sha256"]
            or any(file_sha256(Path(identity[key]["head"])) != identity[key]["sha256"]
                   for key in ("official_contextual", "contextual_diagnostic", "selected_global"))):
        raise ValueError("Replay inputs or source changed")


@torch.no_grad()
def gold_scores(unary, pair, slots, selected):
    """Stable log-marginals, and exact joint NLL only for covered blocks."""
    lengths = chain._validate(unary, pair, None)
    if (unary.shape[0] != 1 or selected.shape != unary.shape[:2]
            or selected.dtype != torch.bool or slots.shape != selected.shape
            or slots.dtype != torch.long or bool(((slots < -1) | (slots >= unary.shape[-1])).any())
            or not bool(selected.any())):
        raise ValueError("Score one nonempty selected block at a time")
    alpha, logz = chain._forward(unary, pair, lengths)
    beta = chain._backward(unary, pair, lengths)
    log_nodes, _ = chain._normalize(alpha + beta)
    positions = selected[0].nonzero().flatten().tolist()
    individual = [None if slots[0, pos] < 0 else -float(log_nodes[0, pos, slots[0, pos]])
                  for pos in positions]
    if any(value is not None and (not math.isfinite(value) or value < -1e-4)
           for value in individual):
        raise FloatingPointError("Invalid in-support singleton log probability")
    covered = all(value is not None for value in individual)
    joint = None
    if covered:
        denied = selected[..., None] & torch.arange(unary.shape[-1], device=unary.device).ne(slots[..., None])
        joint = float((logz - chain.log_partition(unary.masked_fill(denied, -torch.inf), pair)).sum())
        if not math.isfinite(joint) or joint < -1e-4:
            raise FloatingPointError("Invalid covered-block log probability")
    return dict(joint_gold_nll=joint, joint_nll_status="finite" if covered else "infinite_gold_out_of_support",
                singleton_gold_nll=individual,
                sum_singleton_gold_nll=sum(individual) if covered else None,
                supported_singleton_nll_sum=sum(value for value in individual if value is not None))


@torch.no_grad()
def score_state(output, current, selected, gold, potential, global_head, contextual_head,
                *, zero_head=None, count_table=None, mask_id=MASK_ID):
    if (current.ndim != 2 or current.shape[0] != 1 or gold.shape != current.shape
            or selected.shape != current.shape or selected.dtype != torch.bool
            or not bool(selected.any()) or bool((selected & current.ne(mask_id)).any())):
        raise ValueError("Selected positions must be nonempty masked target sites")
    candidates = extract_compact_candidates(output.log_probs, current, mask_id=mask_id,
        nucleus_p=PROTOCOL["nucleus_p"], top_k=PROTOCOL["top_k"])
    ids = candidates.candidate_ids
    found = ids.eq(gold[..., None]) & torch.isfinite(candidates.log_unary)
    slots = torch.where(found.any(-1), found.long().argmax(-1), -1)
    positions = selected[0].nonzero().flatten().tolist()
    covered_positions = [pos for pos in positions if slots[0, pos] >= 0]
    if count_table is None:
        count = count_pair_scores(candidates, potential)
    else:
        if (count_table.shape != (potential.vocab_size, potential.vocab_size)
                or count_table.dtype != torch.float32 or count_table.device != current.device
                or count_table.requires_grad):
            raise ValueError("Invalid frozen count score table")
        safe = ids.masked_fill(ids.eq(potential.vocab_size), 0)
        count = count_table[safe[:, :-1, :, None], safe[:, 1:, None, :]]
    factors = global_head(ids, output.hidden)
    global_pair = compose_pair_scores(count, global_factors=factors, global_strength=1.)
    temperature = PROTOCOL["temperature"]
    # Match the frozen objective's FP32 score construction; the reference DP
    # accumulates partition offsets in FP64 and keeps marginals in log space.
    unary = candidates.log_unary.float() / temperature
    metrics = {"global": gold_scores(unary, global_pair / temperature, slots, selected)}
    contextual_pair = compose_pair_scores(count, global_factors=factors, global_strength=1.,
        contextual_factors=contextual_head(ids, output.hidden), contextual_strength=1.)
    metrics["contextual"] = gold_scores(unary, contextual_pair / temperature, slots, selected)
    nested = None
    if zero_head is not None:
        zero_pair = compose_pair_scores(count, global_factors=factors, global_strength=1.,
            contextual_factors=zero_head(ids, output.hidden), contextual_strength=1.)
        if not torch.equal(zero_pair, global_pair):
            raise ValueError("Official zero contextual head does not reproduce the global energy")
        nested = True
    support_hash = hashlib.sha256()
    for tensor in (ids, candidates.log_unary):
        support_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    visible = current.ne(mask_id)
    mismatches = visible & current.ne(gold)
    return dict(tokens=current[0].tolist(), selected_positions=positions,
        supported_positions=covered_positions, selected_tokens=len(positions),
        visible_tokens=int(visible.sum()), visible_reference_mismatches=int(mismatches.sum()),
        visible_reference_mismatch_fraction=float(mismatches.sum() / visible.sum()) if bool(visible.any()) else None,
        early_visible_eos=bool((visible[:, :-1] & current[:, :-1].eq(int(gold[0, -1]))).any()),
        supported_selected_tokens=len(covered_positions),
        all_selected_supported=len(positions) == len(covered_positions),
        candidate_counts=torch.isfinite(candidates.log_unary)[0].sum(-1).tolist(),
        candidate_sha256=support_hash.hexdigest(), support_shared_by_heads=True,
        gold_injected_tokens=0, zero_context_nesting_checked=nested, models=metrics)


@torch.no_grad()
def replay_example(row, stream_index, model, decoder, potential, global_head, contextual_head,
                   *, zero_head=None, count_table=None, mask_id=MASK_ID, eos_id=EOS_ID):
    """Replay the global trajectory; counterfactual scoring never consumes its RNG."""
    device = decoder.device
    gold = torch.tensor([list(row["target_ids"]) + [eos_id]], dtype=torch.long, device=device)
    prefix = torch.tensor([[eos_id] + list(row["source_ids"]) + [eos_id]], device=device)
    if prefix.shape[1] + gold.shape[1] > CONTEXT_LENGTH:
        raise ValueError("Reference length exceeds the backbone context")
    initial = torch.full_like(gold, mask_id)
    states = {}
    counterfactual_calls = 0

    def forward(current, time):
        tokens = torch.cat((prefix, current), -1)
        output = model(tokens, torch.ones_like(tokens, dtype=torch.bool))
        return SimpleNamespace(log_probs=output.log_probs[:, -gold.shape[1]:],
            hidden=output.hidden[:, -gold.shape[1]:], replay_time=time)

    def observe(current, output, selected):
        nonlocal counterfactual_calls
        name = "first_all_masked" if output.replay_time == 1. else "later_generated_visible"
        if name in states or output.replay_time not in (1., .5):
            raise ValueError("Unexpected replay schedule")
        states[name] = score_state(output, current, selected, gold, potential, global_head,
            contextual_head, zero_head=zero_head, count_table=count_table, mask_id=mask_id)
        if name == "first_all_masked":
            if not bool(current.eq(mask_id).all()):
                raise ValueError("First state must be fully masked")
        else:
            gold_visible = torch.where(current.eq(mask_id), current, gold)
            if torch.equal(current, gold_visible):
                counterfactual = output
            else:
                counterfactual = forward(gold_visible, .5)
                counterfactual_calls += 1
            states["later_gold_visible"] = score_state(counterfactual, gold_visible, selected, gold,
                potential, global_head, contextual_head, count_table=count_table, mask_id=mask_id)
    generators = {kind: torch.Generator(device=device).manual_seed(
        stream_seed(f"iwslt-{PROTOCOL['seed']}", stream_index, kind)) for kind in ("reveal", "proposal")}
    result = sample_window(initial, forward, decoder, steps=2,
        reveal_generator=generators["reveal"], proposal_generator=generators["proposal"], observer=observe)
    return dict(id=row["id"], stream_index=stream_index, reference_tokens=gold[0].tolist(),
        source_sha256=digest(row["source_ids"]), reference_length=gold.shape[1],
        rollout_tokens=result.tokens[0].tolist(), states=states,
        counterfactual_forward_calls=counterfactual_calls, **result.stats)


def validate_record(record, row, stream_index):
    gold = list(row["target_ids"]) + [EOS_ID]
    size = len(gold)
    if (record["id"] != row["id"] or record["stream_index"] != stream_index
            or record["source_sha256"] != digest(row["source_ids"])
            or record["reference_tokens"] != gold or record["reference_length"] != size
            or record["scheduled_steps"] != 2 or record["executed_steps"] != 2
            or len(record["reveal_counts"]) != 2 or sum(record["reveal_counts"]) != size
            or record["actual_forward_calls"] != sum(n > 0 for n in record["reveal_counts"])
            or len(record["rollout_tokens"]) != size
            or any(type(x) is not int or not 0 <= x < MASK_ID for x in record["rollout_tokens"])):
        raise ValueError("Replay population, reference length, or schedule changed")
    counts, states = record["reveal_counts"], record["states"]
    expected = ({"first_all_masked"} if counts[0] else set()) | (
        {"later_generated_visible", "later_gold_visible"} if counts[1] else set())
    if set(states) != expected:
        raise ValueError("Missing or unexpected replay states")
    first_positions = states.get("first_all_masked", {}).get("selected_positions", [])
    if (len(first_positions) != counts[0] or first_positions != sorted(set(first_positions))
            or any(type(pos) is not int or not 0 <= pos < size for pos in first_positions)):
        raise ValueError("First-step selection changed")
    for name, state in states.items():
        selected, supported = state["selected_positions"], state["supported_positions"]
        expected_positions = first_positions if name == "first_all_masked" else [
            p for p in range(size) if p not in first_positions]
        current = [MASK_ID] * size
        if name != "first_all_masked":
            for pos in first_positions:
                current[pos] = gold[pos] if name == "later_gold_visible" else record["rollout_tokens"][pos]
        visible = [pos for pos, value in enumerate(current) if value != MASK_ID]
        mismatches = sum(current[pos] != gold[pos] for pos in visible)
        mismatch_fraction = mismatches / len(visible) if visible else None
        if (selected != expected_positions or selected != sorted(set(selected))
                or state["tokens"] != current or supported != sorted(set(supported))
                or not set(supported).issubset(selected)
                or state["selected_tokens"] != len(selected)
                or state["supported_selected_tokens"] != len(supported)
                or state["all_selected_supported"] != (len(selected) == len(supported))
                or state["visible_tokens"] != len(visible)
                or state["visible_reference_mismatches"] != mismatches
                or ((state["visible_reference_mismatch_fraction"] is None) != (mismatch_fraction is None))
                or (mismatch_fraction is not None and abs(state["visible_reference_mismatch_fraction"] - mismatch_fraction) > 1e-7)
                or state["early_visible_eos"] != any(current[pos] == EOS_ID for pos in visible if pos < size - 1)
                or len(state["candidate_counts"]) != size
                or any(type(n) is not int or not 1 <= n <= 500 for n in state["candidate_counts"])
                or not state["support_shared_by_heads"] or state["gold_injected_tokens"] != 0
                or set(state["models"]) != set(MODEL_NAMES)):
            raise ValueError("Replay state, support, or intervention changed")
        if name != "later_gold_visible" and state["zero_context_nesting_checked"] is not True:
            raise ValueError("Missing zero-context nesting check")
        for values in state["models"].values():
            individual = values["singleton_gold_nll"]
            covered = state["all_selected_supported"]
            if (len(individual) != len(selected)
                    or any((value is None) != (pos not in supported) for pos, value in zip(selected, individual))
                    or any(value is not None and (not math.isfinite(value) or value < -1e-4)
                           for value in individual)
                    or abs(values["supported_singleton_nll_sum"] - sum(
                        value for value in individual if value is not None)) > 1e-8
                    or values["joint_nll_status"] != ("finite" if covered else "infinite_gold_out_of_support")
                    or (not covered and (values["joint_gold_nll"] is not None
                                         or values["sum_singleton_gold_nll"] is not None))
                    or (covered and (values["joint_gold_nll"] is None
                        or not math.isfinite(values["joint_gold_nll"])
                        or values["joint_gold_nll"] < -1e-4
                        or abs(values["sum_singleton_gold_nll"] - sum(individual)) > 1e-8))):
                raise ValueError("Invalid original-support gold NLL")
    expected_extra = int("later_generated_visible" in states and
                         states["later_generated_visible"]["visible_reference_mismatches"] > 0)
    if record["counterfactual_forward_calls"] != expected_extra:
        raise ValueError("Counterfactual forwards differ from the paired context intervention")
    return record


def summarize_states(states):
    selected = sum(s["selected_tokens"] for s in states)
    supported = sum(s["supported_selected_tokens"] for s in states)
    covered = [s for s in states if s["all_selected_supported"]]
    covered_tokens = sum(s["selected_tokens"] for s in covered)
    models = {}
    for name in MODEL_NAMES:
        joint = sum(s["models"][name]["joint_gold_nll"] for s in covered)
        singleton = sum(s["models"][name]["sum_singleton_gold_nll"] for s in covered)
        all_supported = sum(s["models"][name]["supported_singleton_nll_sum"] for s in states)
        models[name] = dict(covered_block_joint_nll_sum=joint,
            covered_block_joint_nll_per_token=joint / covered_tokens if covered_tokens else None,
            covered_block_singleton_nll_sum=singleton,
            covered_block_singleton_nll_per_token=singleton / covered_tokens if covered_tokens else None,
            covered_block_dependency_log_gain_per_token=(singleton - joint) / covered_tokens if covered_tokens else None,
            supported_singleton_nll_sum=all_supported,
            supported_singleton_nll_per_token=all_supported / supported if supported else None)
    contrasts = {key: (models["contextual"][key] - models["global"][key])
                 if models["global"][key] is not None else None for key in (
                     "covered_block_joint_nll_per_token", "covered_block_singleton_nll_per_token",
                     "supported_singleton_nll_per_token")}
    return dict(blocks=len(states), selected_tokens=selected, supported_selected_tokens=supported,
        covered_blocks=len(covered), covered_block_selected_tokens=covered_tokens,
        visible_tokens=sum(s["visible_tokens"] for s in states),
        visible_reference_mismatches=sum(s["visible_reference_mismatches"] for s in states),
        blocks_without_visible_tokens=sum(s["visible_tokens"] == 0 for s in states),
        blocks_with_early_visible_eos=sum(s["early_visible_eos"] for s in states),
        token_coverage=supported / selected if selected else None,
        complete_block_coverage=len(covered) / len(states) if states else None,
        unconditional_joint_nll_status="infinite_gold_out_of_support" if len(covered) != len(states) else "finite",
        models=models, contextual_minus_global=contrasts)


def aggregate(records):
    by_state = {name: summarize_states([r["states"][name] for r in records if name in r["states"]])
                for name in STATE_NAMES}
    paired = [r for r in records if "later_generated_visible" in r["states"]]
    common = [r for r in paired if all(r["states"][name]["all_selected_supported"]
              for name in ("later_generated_visible", "later_gold_visible"))]
    matched = {name: summarize_states([r["states"][name] for r in common])
               for name in ("later_generated_visible", "later_gold_visible")}
    contrast_keys = ("covered_block_joint_nll_per_token", "covered_block_singleton_nll_per_token")
    difference = {key: (matched["later_generated_visible"]["contextual_minus_global"][key]
                       - matched["later_gold_visible"]["contextual_minus_global"][key]) if common else None
                  for key in contrast_keys}
    token_positions, token_sums = [], {state: {model: 0. for model in MODEL_NAMES}
                                     for state in ("later_generated_visible", "later_gold_visible")}
    for record in paired:
        generated, gold = (record["states"][name] for name in token_sums)
        if generated["selected_positions"] != gold["selected_positions"]:
            raise ValueError("Paired contexts must score the same selected positions")
        common_positions = sorted(set(generated["supported_positions"]) & set(gold["supported_positions"]))
        token_positions.append(dict(id=record["id"], positions=common_positions))
        offsets = {pos: index for index, pos in enumerate(generated["selected_positions"])}
        for name in token_sums:
            for model in MODEL_NAMES:
                token_sums[name][model] += sum(record["states"][name]["models"][model]["singleton_gold_nll"][offsets[pos]]
                                              for pos in common_positions)
    retained_tokens = sum(len(row["positions"]) for row in token_positions)
    token_means = {name: {model: value / retained_tokens if retained_tokens else None
                         for model, value in sums.items()} for name, sums in token_sums.items()}
    token_contrasts = {name: (means["contextual"] - means["global"]) if retained_tokens else None
                       for name, means in token_means.items()}
    token_difference = (token_contrasts["later_generated_visible"] - token_contrasts["later_gold_visible"]
                        if retained_tokens else None)
    return dict(examples=len(records), states=by_state,
        later_common_supported=dict(eligible_blocks=len(paired), retained_blocks=len(common),
            eligible_selected_tokens=sum(r["states"]["later_generated_visible"]["selected_tokens"] for r in paired),
            retained_selected_tokens=sum(r["states"]["later_generated_visible"]["selected_tokens"] for r in common),
            retained_ids=[r["id"] for r in common], states=matched,
            contextual_head_contrast_generated_minus_gold=difference),
        later_common_supported_tokens=dict(eligible_blocks=len(paired),
            eligible_selected_tokens=sum(r["states"]["later_generated_visible"]["selected_tokens"] for r in paired),
            retained_tokens=retained_tokens, retained_blocks=sum(bool(row["positions"]) for row in token_positions),
            paired_positions=token_positions, singleton_nll_sums=token_sums,
            singleton_nll_per_token=token_means, contextual_minus_global=token_contrasts,
            contextual_head_contrast_generated_minus_gold=token_difference),
        empty_first_reveal=sum(r["reveal_counts"][0] == 0 for r in records),
        empty_later_reveal=sum(r["reveal_counts"][1] == 0 for r in records))


def record_path(args, position):
    return args.output / "records" / f"{position:05d}.json"


def load_records(args, state):
    expected = {f"{state.positions[row['id']]:05d}.json" for row in state.rows}
    directory = args.output / "records"
    if {p.name for p in directory.glob("*.json")} - expected:
        raise ValueError("Unregistered replay record")
    records = {}
    for row in state.rows:
        position = state.positions[row["id"]]
        path = record_path(args, position)
        if path.exists():
            document = read(path)
            if document["identity_sha256"] != digest(state.identity) or document["record_sha256"] != digest(document["record"]):
                raise ValueError("Replay record identity/hash changed")
            records[row["id"]] = validate_record(document["record"], row, position)
    return records


def publish_summary(args, state, records):
    ordered = [records[row["id"]] for row in state.rows if row["id"] in records]
    result = dict(identity=state.identity, completed=len(ordered), expected=EXAMPLES,
                  complete=len(ordered) == EXAMPLES, summary=aggregate(ordered),
                  outputs={str(record_path(args, state.positions[row["id"]]).relative_to(args.output)):
                           file_sha256(record_path(args, state.positions[row["id"]]))
                           for row in state.rows if row["id"] in records})
    atomic_json(args.output / "readout.json", result)
    return result


def run(args):
    from additive_chain.counts import SparseCountPotential
    from additive_chain.generation import ChainDecoder
    from additive_chain.iwslt_residual_assets import dense_count_scores
    from additive_chain.iwslt_run import log, run_lock
    from additive_chain.translation import load_conditional
    if type(args.max_new_examples) is not int or args.max_new_examples < 0:
        raise ValueError("max-new-examples must be nonnegative")
    state = prepare(args)
    with run_lock(args.output):
        manifest = args.output / "manifest.json"
        if manifest.exists() and read(manifest) != state.identity:
            raise ValueError("Replay manifest changed")
        if not manifest.exists():
            if any(p.name != "run.lock" for p in args.output.iterdir()):
                raise ValueError("Replay output lacks its manifest")
            atomic_json(manifest, state.identity)
        (args.output / "records").mkdir(exist_ok=True)
        records = load_records(args, state)
        if args.command == "readout" and len(records) != EXAMPLES:
            raise ValueError("Replay readout requires all fixed development examples")
        pending = [row for row in state.rows if row["id"] not in records]
        if args.command == "run" and pending:
            model, _ = load_conditional(args.checkpoint, args.tokenizer)
            trained = torch.load(args.trained, map_location="cuda", weights_only=False)
            model.load_state_dict(trained["model"])
            del trained
            model.eval().requires_grad_(False)
            inherited, head, zero = (x.cuda() for x in (state.inherited, state.head, state.zero))
            counts = torch.load(args.data / "counts.pt", weights_only=True)
            potential = SparseCountPotential.from_counts(*(counts[key].cuda() for key in
                ("left_counts", "right_counts", "pair_keys", "pair_counts")), strength=.2, smoothing=.1)
            # This table uses untempered count scores; all energies are divided once in score_state.
            count_table = dense_count_scores(potential)
            decoder = ChainDecoder(mask_id=MASK_ID, device="cuda", count_statistics=counts,
                count_strength=.2, top_k=500, nucleus_p=.9, fast_candidates=True,
                sampling="joint", temperature=.85, global_head=inherited, global_strength=1.)
            try:
                decoder.prepare()
                for index, row in enumerate(pending):
                    if args.max_new_examples and index >= args.max_new_examples:
                        break
                    position = state.positions[row["id"]]
                    record = replay_example(row, position, model, decoder, potential, inherited, head,
                                            zero_head=zero, count_table=count_table)
                    validate_record(record, row, position)
                    atomic_json(record_path(args, position), dict(identity_sha256=digest(state.identity),
                        record_sha256=digest(record), record=record))
                    records[row["id"]] = record
                    log("context_replay_progress", completed=len(records), expected=EXAMPLES)
            finally:
                decoder.close()
        unchanged(args, state)
        publish_summary(args, state, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "readout"))
    for name in ("trained", "data", "length-head", "selection", "global-head", "global-selection",
                 "contextual-selection", "official-evaluation-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-new-examples", type=int, default=0)
    args = parser.parse_args()
    if args.command == "run" and (args.checkpoint is None or args.tokenizer is None):
        parser.error("run requires --checkpoint and --tokenizer")
    run(args)


if __name__ == "__main__":
    main()
