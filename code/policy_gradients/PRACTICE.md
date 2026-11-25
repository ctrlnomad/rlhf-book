# Policy gradient practice

This branch (`practice/policy-gradients`) has the algorithm-specific code removed so you can
write it yourself. Everything else (generation, model loading, the training loop, rewards)
still works. Each removed function raises `NotImplementedError` and its docstring says what
to compute, the tensor shapes, and the conventions the checker expects.
The derivations are in **Chapter 6: Policy Gradient Methods** of the book.

## Exercises

| # | Function | File | Used by |
|---|----------|------|---------|
| 1 | `approx_kl1`, `approx_kl2`, `approx_kl3` | `loss.py` | any run with `beta > 0` |
| 2 | `ReinforceLoss.forward` | `loss.py` | REINFORCE, RLOO |
| 3 | `compute_loo_advantages` | `utils.py` | RLOO |
| 4 | `apply_reward_kl` | `utils.py` | REINFORCE / RLOO / PPO with `beta > 0` |
| 5 | `compute_standardized_advantages` | `utils.py` | GRPO, GSPO, CISPO, SAPO, DAPO |
| 6 | `GRPOLoss.forward` | `loss.py` | GRPO, Dr. GRPO |
| 7 | `compute_nonstandardized_advantages` | `utils.py` | Dr. GRPO |
| 8 | `GSPOLoss.forward` | `loss.py` | GSPO |
| 9 | `CISPOLoss.forward` | `loss.py` | CISPO |
| 10 | `SAPOLoss.forward` | `loss.py` | SAPO |
| 11 | `DAPOLoss.forward`, `dapo_length_penalty` (in `_response_penalties`), `RolloutEngine._dapo_filter` | `loss.py`, `utils.py`, `rollout.py` | DAPO |
| 12 | `compute_maxrl_advantages`, `MaxRLLoss.forward` | `utils.py`, `loss.py` | MaxRL |
| 13 | `compute_gae`, `PPOLoss.forward` | `utils.py`, `loss.py` | PPO (hardest) |

## Checking your work

From `code/`:

```bash
uv run --extra dev pytest tests/test_practice_pg.py -q            # all exercises
uv run --extra dev pytest tests/test_practice_pg.py -q -k grpo     # just GRPO
uv run --extra dev pytest tests/test_practice_pg.py -q -k "kl_"    # just the KL estimators
```

The checker runs your code on fixed inputs and compares it with outputs recorded from the
original implementation (`tests/data/practice_pg_reference.pt`). Losses are compared on the
value **and** the gradient, so a missing stop-gradient or a wrong mask fails even if the
number looks right. `tests/test_kl3_stability.py` and `tests/test_binary_reward.py` also
start passing again once the KL estimators and MaxRL advantages are in.

Then try it for real with a short run, e.g. 10 steps of GRPO:

```bash
uv run python -m policy_gradients.train --config policy_gradients/configs/grpo_test.yaml
```

## Peeking

The original code is on `main`. `git diff main -- policy_gradients/loss.py` shows the answers.
