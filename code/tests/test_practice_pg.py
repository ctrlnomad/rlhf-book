"""Checker for the policy-gradient practice exercises.

Each case runs YOUR implementation on fixed inputs and compares it with the output
recorded from the original implementation (tests/data/practice_pg_reference.pt).
Losses are checked on both the loss value and the gradient, so getting a
stop-gradient or a mask wrong will show up even when the number looks right.

    uv run pytest tests/test_practice_pg.py -q            # everything
    uv run pytest tests/test_practice_pg.py -q -k grpo     # one algorithm
    uv run pytest tests/test_practice_pg.py -q -k "kl_"    # KL estimators only

Suggested order: kl -> reinforce/rloo -> grpo/drgrpo -> gspo -> cispo -> sapo
-> dapo -> maxrl -> gae/ppo (hardest last).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from policy_gradients import loss as L
from policy_gradients import utils as U
from policy_gradients.buffer import Experience
from policy_gradients.rollout import RolloutEngine


REF_PATH = Path(__file__).parent / "data" / "practice_pg_reference.pt"
CASES = torch.load(REF_PATH, weights_only=False)
ATOL, RTOL = 1e-5, 1e-4


def _close(actual, expected, what: str) -> None:
    assert actual is not None, f"{what}: got None"
    actual = torch.as_tensor(actual).detach().float()
    expected = expected.float()
    assert actual.shape == expected.shape, (
        f"{what}: shape {tuple(actual.shape)} != expected {tuple(expected.shape)}"
    )
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = (actual - expected).abs()
        raise AssertionError(
            f"{what}: max abs diff {diff.max().item():.3e}\nexpected:\n{expected}\ngot:\n{actual}"
        )


def _run(case: dict) -> None:
    kind, inp = case["kind"], case["inputs"]
    if kind == "kl":
        out = getattr(L, case["fn"])(
            inp["log_probs"].clone(), inp["log_probs_ref"].clone(), inp["action_mask"]
        )
        _close(out, case["output"], case["fn"])
    elif kind == "adv":
        out = getattr(U, case["fn"])(*[v.clone() for v in inp.values()])
        _close(out, case["output"], case["fn"])
    elif kind == "gae":
        out = U.compute_gae(
            inp["rewards"].clone(),
            inp["action_mask"].clone(),
            inp["values"].clone(),
            inp["gamma"],
            inp["lam"],
        )
        _close(out, case["output"], "compute_gae")
    elif kind == "reward_kl":
        rw = TensorDict({"total": inp["total"].clone()}, batch_size=[inp["total"].shape[0]])
        out = U.apply_reward_kl(
            rw,
            inp["log_probs"],
            inp["log_probs_ref"],
            inp["action_mask"],
            SimpleNamespace(**case["cfg"]),
        )
        _close(out["total"], case["output"], "apply_reward_kl()['total']")
    elif kind == "penalty":
        out = U._response_penalties(list(inp["lengths"]), SimpleNamespace(**case["cfg"]))
        _close(torch.tensor(out, dtype=torch.float32), case["output"], "_response_penalties")
    elif kind == "filter":
        c = inp["correctness"]
        n = c.shape[0]
        exp = Experience(
            sequence_ids=torch.zeros(n, 3, dtype=torch.long),
            attention_mask=torch.ones(n, 3),
            action_mask=torch.ones(n, 2, dtype=torch.bool),
            rewards=TensorDict({"correctness": c.clone()}, batch_size=[n]),
        )
        eng = object.__new__(RolloutEngine)
        eng.cfg = SimpleNamespace(**case["cfg"])
        kept = eng._dapo_filter(exp) is not None
        want = bool(case["output"])
        assert kept == want, (
            f"_dapo_filter on correctness {c.squeeze(-1).tolist()}: "
            f"{'kept' if kept else 'dropped'} the group, expected it to be {'kept' if want else 'dropped'}"
        )
    elif kind == "loss":
        lp = inp["log_probs"].clone().requires_grad_(True)
        vals = inp["values"].clone().requires_grad_(True) if inp["values"] is not None else None
        B, S = lp.shape
        exp = Experience(
            sequence_ids=torch.zeros(B, S + 1, dtype=torch.long),
            attention_mask=torch.ones(B, S + 1),
            action_mask=inp["action_mask"],
            advantages=inp["advantages"],
            log_probs_old=inp["log_probs_old"],
            log_probs_ref=inp["log_probs_ref"],
            values_old=inp["values_old"],
        )
        objective = U.get_loss_objective(loss=case["loss"], **case["kwargs"])
        out = objective(log_probs=lp, experience=exp, values=vals)
        _close(out, case["output"], f"{case['loss']} loss value")
        out.backward()
        _close(lp.grad, case["grad_log_probs"], f"{case['loss']} d(loss)/d(log_probs)")
        if case["grad_values"] is not None:
            _close(vals.grad, case["grad_values"], f"{case['loss']} d(loss)/d(values)")
    else:  # pragma: no cover
        raise ValueError(kind)


@pytest.mark.parametrize("name", list(CASES))
def test_case(name: str) -> None:
    _run(CASES[name])
