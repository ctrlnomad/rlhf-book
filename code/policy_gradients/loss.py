# Policy Gradient Loss Functions
#
# Original implementation by Zafir Stojanovski (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0
#
# Implements:
# - REINFORCE (Williams, 1992)
# - PPO (Schulman et al., 2017)
# - GRPO (Shao et al., 2024)
# - GSPO (Zheng et al., 2025)
# - CISPO (MiniMax, 2025)
# - SAPO (Qwen Team, 2025)
# - DAPO (ByteDance Team, 2025)
# - MaxRL (Tajwar et al., 2026)

import torch
import torch.nn as nn

from .buffer import Experience


def approx_kl3(
    log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """TODO: k3 estimator of KL(pi || pi_ref) for tokens sampled from pi.

    With r = pi_ref / pi, the estimator is  r - log(r) - 1.
    Use torch.expm1 rather than exp(...) - 1 for numerical stability near r = 1
    (tests/test_kl3_stability.py checks this).
    Masked-out positions must come out as exactly 0: zero the log-ratio with
    action_mask (when it is not None) before applying the formula.
    Returns (B, S).
    See: http://joschu.net/blog/kl-approx.html
    """
    raise NotImplementedError("TODO: implement approx_kl3")


def approx_kl2(
    log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """TODO: k2 estimator of KL(pi || pi_ref):  (log(pi / pi_ref))^2 / 2.

    Zero the log-ratio with action_mask (when it is not None). Returns (B, S).
    See: http://joschu.net/blog/kl-approx.html
    """
    raise NotImplementedError("TODO: implement approx_kl2")


def approx_kl1(
    log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """TODO: k1 estimator of KL(pi || pi_ref):  log(pi / pi_ref).

    Samples come from pi, so the ratio is policy over reference
    (tests/test_kl3_stability.py checks the sign).
    Zero the log-ratio with action_mask (when it is not None). Returns (B, S).
    See: http://joschu.net/blog/kl-approx.html
    """
    raise NotImplementedError("TODO: implement approx_kl1")


def get_approx_kl(
    kl_estimator: str,
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute KL approximation using the selected estimator."""
    if kl_estimator == "kl1":
        return approx_kl1(log_probs, log_probs_ref, action_mask)
    if kl_estimator == "kl2":
        return approx_kl2(log_probs, log_probs_ref, action_mask)
    if kl_estimator == "kl3":
        return approx_kl3(log_probs, log_probs_ref, action_mask)
    raise ValueError(f"Unsupported kl_estimator: {kl_estimator}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int | None = None,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean over masked positions."""
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    return (tensor * mask).sum(dim=dim, keepdim=keepdim) / mask.sum(
        dim=dim, keepdim=keepdim
    ).clamp_min(eps)


class GRPOLoss(nn.Module):
    """Group Relative Policy Optimization loss (Shao et al., 2024).

    GRPO uses group-level advantage normalization and clipped ratio updates.
    See Chapter 6 of RLHF Book for mathematical derivation.
    """

    def __init__(
        self, clip_eps_lo: float, clip_eps_hi: float, beta: float, kl_estimator: str, **kwargs
    ) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.kl_estimator = kl_estimator

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: GRPO loss (also used for Dr. GRPO; they differ only in the advantages).

        experience.advantages: (B, 1), one scalar per completion, broadcast to every token.
        Per token: importance ratio pi / pi_old (from log_probs and
        experience.log_probs_old), then the PPO-style clipped surrogate with
        clip range [1 - clip_eps_lo, 1 + clip_eps_hi]. Remember the loss is the
        NEGATIVE of the objective you want to maximise.

        Optional KL penalty: if self.beta > 0, add self.beta * (per-token KL estimate from
        get_approx_kl(self.kl_estimator, log_probs, experience.log_probs_ref, action_mask))
        to the per-token loss before reducing.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement GRPOLoss.forward")


class GSPOLoss(nn.Module):
    """Group-Sequence Policy Optimization loss (Zheng et al., 2025).

    GSPO applies clipping to sequence-level log probability ratios.
    """

    def __init__(
        self, clip_eps_lo: float, clip_eps_hi: float, beta: float, kl_estimator: str, **kwargs
    ) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.kl_estimator = kl_estimator

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: GSPO loss: like GRPO, but the importance ratio is per SEQUENCE.

        Sequence ratio = exp(mean over completion tokens of (log_probs - log_probs_old)),
        i.e. the geometric mean of the token ratios, shape (B, 1).
        Clip and form the surrogate with that sequence ratio and the (B, 1) advantages,
        the same way GRPO does per token.

        Optional KL penalty: if self.beta > 0, add self.beta * (per-token KL estimate from
        get_approx_kl(self.kl_estimator, log_probs, experience.log_probs_ref, action_mask))
        to the per-token loss before reducing.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement GSPOLoss.forward")


class ReinforceLoss(nn.Module):
    """REINFORCE loss (Williams, 1992).

    The classic policy gradient: -log(pi) * advantage
    See Chapter 6 of RLHF Book for derivation from the policy gradient theorem.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: REINFORCE loss (also used for RLOO; they differ only in the advantages).

        The policy-gradient surrogate whose gradient is -A * grad log pi.
        experience.advantages is (B, 1): the reward (REINFORCE) or the
        leave-one-out advantage (RLOO). No clipping, no ratio.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement ReinforceLoss.forward")


class MaxRLLoss(nn.Module):
    """MaxRL loss (Tajwar et al., 2026).

    Uses binary rewards per completion:
        r = 1 if correctness == 1 and format == 1 else 0
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: MaxRL loss.

        Same functional form as REINFORCE; what makes MaxRL different lives in
        compute_maxrl_advantages (utils.py). experience.advantages is (B, 1).

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement MaxRLLoss.forward")


class CISPOLoss(nn.Module):
    """Clipped Importance Sampling Policy Optimization loss (MiniMax, 2025).

    CISPO uses stop-gradient on the clipped ratio, providing a different
    gradient signal than PPO/GRPO.
    """

    def __init__(
        self, clip_eps_lo: float, clip_eps_hi: float, beta: float, kl_estimator: str, **kwargs
    ) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.beta = beta
        self.kl_estimator = kl_estimator

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: CISPO loss.

        Compute the token ratio pi / pi_old and clip it to
        [1 - clip_eps_lo, 1 + clip_eps_hi], but treat the clipped ratio as a
        CONSTANT weight (no gradient flows through it). Multiply it by the
        advantage and by log_probs, REINFORCE-style. Unlike PPO/GRPO, clipped
        tokens still get a gradient. experience.advantages is (B, 1).
        The checker compares gradients, so a missing stop-gradient will fail.

        Optional KL penalty: if self.beta > 0, add self.beta * (per-token KL estimate from
        get_approx_kl(self.kl_estimator, log_probs, experience.log_probs_ref, action_mask))
        to the per-token loss before reducing.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement CISPOLoss.forward")


class SAPOLoss(nn.Module):
    """Soft Adaptive Policy Optimization loss (Qwen Team, 2025).

    Replaces hard clipping with a smooth sigmoid gate that continuously
    attenuates gradients as tokens move off-policy.
    See Chapter 6 - Further Reading of the RLHF Book
    """

    def __init__(
        self, sapo_temp_pos: float, sapo_temp_neg: float, beta: float, kl_estimator: str, **kwargs
    ) -> None:
        super().__init__()
        self.sapo_temp_pos = sapo_temp_pos
        self.sapo_temp_neg = sapo_temp_neg
        self.beta = beta
        self.kl_estimator = kl_estimator

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: SAPO loss: replace hard clipping with a smooth sigmoid gate.

        Per token, with ratio r = pi / pi_old and advantage A ((B, 1), broadcast):
          tau  = self.sapo_temp_pos where A > 0, else self.sapo_temp_neg
          gate = sigmoid(tau * (r - 1)) * 4 / tau
          per-token policy loss = -gate * A
        (At r = 1 the gate's derivative is 1, so on-policy it matches the
        vanilla policy gradient; it fades out as r moves away from 1.)

        Optional KL penalty: if self.beta > 0, add self.beta * (per-token KL estimate from
        get_approx_kl(self.kl_estimator, log_probs, experience.log_probs_ref, action_mask))
        to the per-token loss before reducing.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement SAPOLoss.forward")


class PPOLoss(nn.Module):
    """Proximal Policy Optimization loss (Schulman et al., 2017).

    PPO combines clipped policy updates with a value function baseline.
    See Chapter 6 of RLHF Book for the full derivation.
    """

    def __init__(
        self,
        clip_eps_lo: float,
        clip_eps_hi: float,
        clip_eps_val: float,
        vf_coef: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi
        self.clip_eps_val = clip_eps_val
        self.vf_coef = vf_coef

    def forward(
        self, log_probs: torch.Tensor, experience: Experience, values: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """TODO: PPO loss = clipped policy surrogate + vf_coef * clipped value loss.

        Here experience.advantages is PER TOKEN, shape (B, S) (from GAE), and
        values / experience.values_old are (B, S) value-head outputs.
        Policy part: same clipped surrogate as GRPO, with
          [1 - clip_eps_lo, 1 + clip_eps_hi].
        Value part:
          returns = advantages + values_old          (since A = G - V)
          clip the new values to within +-clip_eps_val of values_old
          value loss per token = 0.5 * max(squared error unclipped,
                                           squared error clipped)  against returns
        Combine per token: policy + self.vf_coef * value, then reduce. No KL term here
        (PPO's KL penalty goes into the reward, see apply_reward_kl in utils.py).
        The checker also compares d(loss)/d(values).

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False

        Reduction convention used throughout this repo (the checker expects it):
          per-token loss (B, S) -> masked_mean(..., mask=action_mask, dim=-1) -> (B,) -> .mean(dim=0)
        i.e. average over each sequence's completion tokens, then over sequences.
        `masked_mean` and `get_approx_kl` at the top of this file are there for you to use.
        Return a scalar tensor. The checker compares the loss value AND d(loss)/d(log_probs).
        """
        raise NotImplementedError("TODO: implement PPOLoss.forward")


class DAPOLoss(nn.Module):
    """Decoupled Clip and Dynamic sAmpling Policy Optimization loss (ByteDance, 2025).

    DAPO uses token-level loss normalization, decoupled lower/upper clipping bounds
    (Clip-Higher), and no KL penalty. See Section 3 of the DAPO paper.
    """

    def __init__(self, clip_eps_lo: float, clip_eps_hi: float, **kwargs) -> None:
        super().__init__()
        self.clip_eps_lo = clip_eps_lo
        self.clip_eps_hi = clip_eps_hi

    def forward(self, log_probs: torch.Tensor, experience: Experience, **kwargs) -> torch.Tensor:
        """TODO: DAPO loss: GRPO's clipped surrogate with two changes.

        1. Clip-higher: the range is [1 - clip_eps_lo, 1 + clip_eps_hi] with
           clip_eps_hi > clip_eps_lo (0.2 / 0.28 in the config). Same code as GRPO.
        2. Token-level normalisation: NOT the repo's usual per-sequence mean.
           Sum the per-token loss over ALL completion tokens in the batch and
           divide by the total number of completion tokens (clamp the count to at
           least 1e-8). Long sequences therefore weigh more.
        No KL term. experience.advantages is (B, 1).
        Return a scalar tensor.

        Shapes (B = sequences in the batch, S = sequence length - 1):
          log_probs, log_probs_old, log_probs_ref: (B, S) float, log-prob of each next token
          action_mask: (B, S) bool, True only on generated (completion) tokens;
                       prompt tokens (left) and padding are False
        """
        raise NotImplementedError("TODO: implement DAPOLoss.forward")
