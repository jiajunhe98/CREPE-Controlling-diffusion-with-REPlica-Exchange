"""
Sequential Monte Carlo (SMC) weight computation for discrete diffusion models.

This module implements importance sampling weights for unbiased guidance in discrete
diffusion models, following the mathematical framework from "Debiasing Guidance for
Discrete Diffusion with Sequential Monte Carlo" by Brian Lee et al.

The core idea is to use SMC to sample from tempered distributions:
    p_t^α(x_t) ∝ p_t(x_t) p_t(ζ | x_t)^α

where α is the SMC temperature parameter and ζ is the conditioning variable.
"""

import abc
from typing import Callable, List, Tuple

import torch
from jaxtyping import Bool, Float, Int

import lib.models.utils as mutils

# TODO: This module is actually not great. Its abstraction expects that we compute the uncon transition, proposal transition and conditional likelihood ratio. But actually the job of this is to compute the log-weights update. And we should clarify the abstraction to make it clear that this is our only objective. This would relax the expected input of compute_weight_update and give more flexibility.


class SMCWeightComputer(abc.ABC):
    """Abstract base class for computing SMC importance weights.

    This class defines the interface for computing the three key quantities
    needed for SMC importance sampling:
    1. log p_{t_{j+1}|t_j}(Y_{t_{j+1}} | Y_{t_j}) - unconditional transition
    2. log q_{t_{j+1}|t_j}(Y_{t_{j+1}} | Y_{t_j}) - proposal transition
    3. log [p_t(ζ|Y_{t_{j+1}}) / p_t(ζ|Y_{t_j})]^α - conditional likelihood ratio
    """

    def __init__(self, smc_temperature: float = 1.0):
        """Initialize the weight computer.

        Args:
            smc_temperature: Temperature α for weighting the conditional likelihood ratio.
        """
        self.smc_temperature = smc_temperature

    @abc.abstractmethod
    def compute_weight_update(
        self,
        x_prev: Int[
            torch.Tensor, "batch datadim"
        ],  # Previous particle states: [batch, datadim]
        x_curr: Int[
            torch.Tensor, "batch datadim"
        ],  # Current particle states: [batch, datadim]
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for sequences
        proposal_log_prob: Float[
            torch.Tensor, "batch datadim"
        ],  # Log q(x_curr|x_prev): [batch, datadim]
        uncond_log_prob: Float[
            torch.Tensor, "batch datadim"
        ],  # Log p(x_curr|x_prev): [batch, datadim]
        cond_log_ratio: Float[
            torch.Tensor, "batch datadim"
        ],  # Log [p(ζ|x_curr)/p(ζ|x_prev)]: [batch, datadim]
    ) -> Float[torch.Tensor, "batch"]:
        """Compute the log weight update for one SMC step.

        Args:
            x_prev: Previous particle states [batch, datadim]
            x_curr: Current particle states [batch, datadim]
            cond: Conditioning variable (e.g., class labels)
            proposal_log_prob: log q_{t_{j+1}|t_j}(x_curr | x_prev) [batch, datadim]
            uncond_log_prob: log p_{t_{j+1}|t_j}(x_curr | x_prev) [batch, datadim]
            cond_log_ratio: log [p_t(ζ|x_curr) / p_t(ζ|x_prev)] [batch, datadim]

        Returns:
            Log weight update for each particle [batch]
        """
        pass


class CFGWeightComputer(SMCWeightComputer):
    """SMC weight computation for Classifier-Free Guidance.

    This class implements the weight update formula for CFG:

    log W_{k+1} = log W_k +
                  log p_{t_{j+1}|t_j}(Y_{t_{j+1}} | Y_{t_j}) -
                  log q_{t_{j+1}|t_j}(Y_{t_{j+1}} | Y_{t_j}) +
                  α * log [p_t(ζ|Y_{t_{j+1}}) / p_t(ζ|Y_{t_j})]

    The conditional likelihood ratio is computed using Bayes' rule:
    p_t(ζ|y) / p_t(ζ|x) = [p_t(y|ζ) / p_t(x|ζ)] * [p_t(x) / p_t(y)]

    Both ratios are available from conditional and unconditional models.
    """

    def compute_weight_update(
        self,
        x_prev: Int[
            torch.Tensor, "batch datadim"
        ],  # Previous particle states: [batch, datadim]
        x_curr: Int[
            torch.Tensor, "batch datadim"
        ],  # Current particle states: [batch, datadim]
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for sequences
        proposal_log_prob: Float[
            torch.Tensor, "batch datadim"
        ],  # Log q(x_curr|x_prev): [batch, datadim]
        uncond_log_prob: Float[
            torch.Tensor, "batch datadim"
        ],  # Log p(x_curr|x_prev): [batch, datadim]
        cond_log_ratio: Float[
            torch.Tensor, "batch datadim"
        ],  # Log [p(ζ|x_curr)/p(ζ|x_prev)]: [batch, datadim]
    ) -> Float[torch.Tensor, "batch"]:
        """Compute CFG weight update.

        The weight update combines three terms:
        1. Importance weight: p/q ratio between unconditional and proposal
        2. Conditional reweighting: α-weighted likelihood ratio

        Args:
            x_prev: Previous particle states [batch, datadim]
            x_curr: Current particle states [batch, datadim]
            cond: Class labels or other conditioning variable
            proposal_log_prob: Log transition probabilities under proposal distribution [batch, datadim]
            uncond_log_prob: Log transition probabilities under unconditional distribution [batch, datadim]
            cond_log_ratio: Log conditional likelihood ratios [batch, datadim]

        Returns:
            Log weight update [batch]
        """
        # Sum over data dimensions for each particle
        # All computations in log space for numerical stability

        # Importance sampling weight: log p - log q
        uncond_term: Float[torch.Tensor, "batch"] = uncond_log_prob.sum(
            dim=-1
        )  # Sum log p over datadim: [batch, datadim] -> [batch]
        proposal_term: Float[torch.Tensor, "batch"] = proposal_log_prob.sum(
            dim=-1
        )  # Sum log q over datadim: [batch, datadim] -> [batch]

        # Conditional likelihood ratio weighted by SMC temperature α
        ratio_term: Float[torch.Tensor, "batch"] = cond_log_ratio.sum(
            dim=-1
        )  # Sum log ratio over datadim: [batch, datadim] -> [batch]
        weighted_ratio: Float[torch.Tensor, "batch"] = (
            self.smc_temperature * ratio_term
        )  # α * sum(log ratio): [batch]

        # SMC weight update formula: log W_{k+1} = log W_k + log p - log q + α * log[ratio]
        log_weight_update: Float[torch.Tensor, "batch"] = (
            uncond_term  # + log p (unconditional transition probability)
            - proposal_term  # - log q (proposal transition probability)
            + weighted_ratio  # + α * log[p(ζ|x_curr)/p(ζ|x_prev)] (conditional reweighting)
        )  # Final weight update per particle: [batch]

        return log_weight_update


# def compute_transition_log_probs(
#     graph: "graph.Graph",  # Rate matrix graph object (Uniform or Absorbing)
#     x_prev: Int[
#         torch.Tensor, "batch datadim"
#     ],  # Previous particle states: [batch, datadim]
#     x_curr: Int[
#         torch.Tensor, "batch datadim"
#     ],  # Current particle states: [batch, datadim]
#     rate: Float[
#         torch.Tensor, "batch datadim vocab"
#     ],  # Rate matrix R*dt: [batch, datadim, vocab]
# ) -> Float[torch.Tensor, "batch datadim"]:
#     """Compute log transition probabilities for CTMC transitions.

#     The transition probability is:
#     p(x_curr | x_prev) = δ(x_curr, x_prev) + rate(x_prev, x_curr) * dt

#     where rate includes the step size dt.

#     Args:
#         graph: Rate matrix graph object
#         x_prev: Previous states [batch, datadim]
#         x_curr: Current states [batch, datadim]
#         rate: Rate matrix entries [batch, datadim, vocab]

#     Returns:
#         Log transition probabilities [batch, datadim]
#     """
#     # Compute CTMC transition probabilities: P(x_curr | x_prev) = δ(x_curr, x_prev) + R(x_prev, x_curr) * dt

#     # Create identity matrix term: δ(x_curr, x_prev)
#     identity_term: Float[torch.Tensor, "batch datadim vocab"] = (
#         torch.nn.functional.one_hot(x_prev, num_classes=graph.dim).to(rate)
#     )  # One-hot encoding: [batch, datadim] -> [batch, datadim, vocab]

#     # Add rate matrix term: δ + R*dt
#     probs: Float[torch.Tensor, "batch datadim vocab"] = (
#         identity_term + rate
#     )  # Element-wise addition: [batch, datadim, vocab]

#     # Extract probabilities for actual transitions x_prev -> x_curr
#     # gather selects probs[b, d, x_curr[b, d]] for each batch b and datadim d
#     transition_probs: Float[torch.Tensor, "batch datadim 1"] = torch.gather(
#         probs, -1, x_curr[..., None]
#     )  # Gather along vocab dimension: [batch, datadim, vocab] -> [batch, datadim, 1]

#     transition_probs: Float[torch.Tensor, "batch datadim"] = transition_probs.squeeze(
#         -1
#     )  # Remove singleton: [batch, datadim, 1] -> [batch, datadim]

#     # Convert to log space with numerical stability (avoid log(0))
#     log_probs: Float[torch.Tensor, "batch datadim"] = torch.clamp(
#         transition_probs, min=1e-8
#     ).log()  # Clamp and log: [batch, datadim] -> [batch, datadim]

#     return log_probs


# def extract_cond_log_ratios(
#     uncond_score: Float[
#         torch.Tensor, "batch datadim vocab"
#     ],  # Unconditional ratios: p_t(x_curr)/p_t(x_prev) [batch, datadim, vocab]
#     cond_score: Float[
#         torch.Tensor, "batch datadim vocab"
#     ],  # Conditional ratios: p_t(x_curr|ζ)/p_t(x_prev|ζ) [batch, datadim, vocab]
#     x_curr: Int[
#         torch.Tensor, "batch datadim"
#     ],  # Current particle states: [batch, datadim]
# ) -> Float[torch.Tensor, "batch datadim"]:
#     """Extract conditional likelihood ratios from CFG scores.

#     The scores are already ratios:
#     - uncond_score[x_curr] = p_t(x_curr) / p_t(x_prev)
#     - cond_score[x_curr] = p_t(x_curr|ζ) / p_t(x_prev|ζ)

#     Using Bayes' rule:
#     p_t(ζ|x_curr) / p_t(ζ|x_prev) = [p_t(x_curr|ζ) / p_t(x_prev|ζ)] / [p_t(x_curr) / p_t(x_prev)]
#                                    = cond_score[x_curr] / uncond_score[x_curr]

#     Args:
#         uncond_score: Unconditional scores [batch, datadim, vocab]
#         cond_score: Conditional scores [batch, datadim, vocab]
#         x_curr: Current states [batch, datadim]

#     Returns:
#         Log conditional likelihood ratios [batch, datadim]
#     """
#     # Extract probability ratios at current positions
#     uncond_ratio: Float[torch.Tensor, "batch datadim"] = torch.gather(
#         uncond_score, -1, x_curr[..., None]
#     ).squeeze(
#         -1
#     )  # p_t(x_curr)/p_t(x_prev): [batch, datadim, vocab] -> [batch, datadim]

#     cond_ratio: Float[torch.Tensor, "batch datadim"] = torch.gather(
#         cond_score, -1, x_curr[..., None]
#     ).squeeze(
#         -1
#     )  # p_t(x_curr|ζ)/p_t(x_prev|ζ): [batch, datadim, vocab] -> [batch, datadim]

#     # Convert to log space and compute log ratio
#     # log [p_t(ζ|x_curr) / p_t(ζ|x_prev)] = log[cond_ratio] - log[uncond_ratio]
#     log_cond_ratio: Float[torch.Tensor, "batch datadim"] = torch.clamp(
#         cond_ratio, min=1e-8
#     ).log()
#     log_uncond_ratio: Float[torch.Tensor, "batch datadim"] = torch.clamp(
#         uncond_ratio, min=1e-8
#     ).log()

#     log_likelihood_ratio: Float[torch.Tensor, "batch datadim"] = (
#         log_cond_ratio - log_uncond_ratio
#     )

#     return log_likelihood_ratio


# def _log_R(
#     t1: Float[torch.Tensor, "batch"],
#     t2: Float[torch.Tensor, "batch"],
#     step_size: float,
#     Y_path: List[Int[torch.Tensor, "batch datadim"]],
#     score_path: List[Float[torch.Tensor, "batch datadim vocab"]],
#     forward_prob_fn: Callable,
#     reverse_prob_fn: Callable,
#     device: torch.device,
# ) -> Float[torch.Tensor, "batch"]:
#     """
#     Function that returns the log of the Radon Nykodym Estimator ratio.
#     log(R(Y_{[t1, t2]})) = sum_{n=1}^{N-1} p_{n|n+1}^Q2 (Yn | Yn+1) -  p_{n+1|n}^Q1 (Yn+1 | Yn)
#     For now we assume Q1 is the forward rate matrix and Q2 is the reverse rate matrix. n is the current time step (t) and n+1 is the next time step (t + step_size).

#     Y_path is a list of tensors of shape (batch_size, datadim). that represents [Y_t1, Y_t1 + dt, Y_t1 + 2dt, ...]
#     forward_prob_fn: (Yn, Yn+1, t, step_size) -> p_{n+1|n}^Q2 (Yn+1 | Yn)
#     reverse_prob_fn: (score_fn, Yn, Yn+1, t, step_size) -> p_{n|n+1}^Q1 (Yn | Yn+1)
#     """
#     assert len(Y_path) >= 2
#     assert len(score_path) == len(Y_path) - 1

#     log_R_forward: Float[torch.Tensor, "batch"] = torch.zeros(
#         Y_path[0].shape[0], device=device
#     )
#     log_R_reverse: Float[torch.Tensor, "batch"] = torch.zeros(
#         Y_path[0].shape[0], device=device
#     )
#     for i in range(len(Y_path) - 1):
#         past_x = Y_path[i]
#         current_x = Y_path[i + 1]
#         forward_prob: Float[torch.Tensor, "batch datadim"] = forward_prob_fn(
#             past_x, current_x, t1, step_size
#         )
#         log_forward_step = forward_prob.log().sum(-1)
#         log_R_forward += log_forward_step

#     for i in range(len(Y_path) - 1, 0, -1):
#         score = score_path[i - 1]
#         current_x = Y_path[i]
#         past_x = Y_path[i - 1]
#         reverse_prob: Float[torch.Tensor, "batch datadim"] = reverse_prob_fn(
#             score, past_x, current_x, t1, step_size
#         )
#         log_reverse_step = reverse_prob.log().sum(-1)
#         log_R_reverse += log_reverse_step

#     return log_R_reverse - log_R_forward


def _log_R_score(
    t1: Float[torch.Tensor, "batch"],
    t2: Float[torch.Tensor, "batch"],
    step_size: float,
    Y_path: List[Int[torch.Tensor, "batch datadim"]],
    score_path: List[Float[torch.Tensor, "batch datadim vocab"]],
    forward_prob_fn: Callable,
    reverse_prob_fn: Callable,
    device: torch.device,
) -> Float[torch.Tensor, "batch"]:
    """
    Alternative implementation that uses scores directly.
    """
    assert len(Y_path) == 2
    assert len(score_path) == 1

    score: Float[torch.Tensor, "batch datadim vocab"] = score_path[0]

    Y_earlier: Int[torch.Tensor, "batch datadim"] = Y_path[0]
    Y_later: Int[torch.Tensor, "batch datadim"] = Y_path[1]

    gathered_scores: Float[torch.Tensor, "batch datadim"] = torch.gather(
        score, -1, Y_earlier[..., None]
    ).squeeze(-1)

    no_transition_mask: Bool[torch.Tensor, "batch datadim"] = Y_earlier == Y_later

    log_gathered_scores: Float[torch.Tensor, "batch datadim"] = gathered_scores.log()

    # Compute individual probability components for debugging
    reverse_probs: Float[torch.Tensor, "batch datadim"] = reverse_prob_fn(
        score, Y_earlier, Y_later, t1, step_size
    )
    forward_probs: Float[torch.Tensor, "batch datadim"] = forward_prob_fn(
        Y_earlier, Y_later, t1, step_size
    )

    transition_log_probs: Float[torch.Tensor, "batch datadim"] = (
        reverse_probs.log() - forward_probs.log()
    )

    final_log_gathered_scores = torch.where(
        no_transition_mask, transition_log_probs, log_gathered_scores
    )

    final_result: Float[torch.Tensor, "batch"] = final_log_gathered_scores.sum(-1)

    # DEBUG: Print comprehensive statistics (comment out this line to disable)
    # print_log_r_debug(
    #     gathered_scores,
    #     log_gathered_scores,
    #     reverse_probs,
    #     forward_probs,
    #     transition_log_probs,
    #     no_transition_mask,
    #     final_log_gathered_scores,
    #     final_result,
    # )

    return final_result


def _log_R_prob(
    t1: Float[torch.Tensor, "batch"],
    t2: Float[torch.Tensor, "batch"],
    step_size: float,
    Y_path: List[Int[torch.Tensor, "batch datadim"]],
    score_path: List[Float[torch.Tensor, "batch datadim vocab"]],
    forward_prob_fn: Callable,
    reverse_prob_fn: Callable,
    device: torch.device,
) -> Float[torch.Tensor, "batch"]:
    """
    Alternative to compute log_R using probabilities.
    """
    assert len(Y_path) == 2
    assert len(score_path) == 1

    score: Float[torch.Tensor, "batch datadim vocab"] = score_path[0]

    Y_earlier: Int[torch.Tensor, "batch datadim"] = Y_path[0]
    Y_later: Int[torch.Tensor, "batch datadim"] = Y_path[1]

    forward_prob: Float[torch.Tensor, "batch"] = forward_prob_fn(
        Y_earlier, Y_later, t1, step_size
    ).sum(-1)
    reverse_prob: Float[torch.Tensor, "batch"] = reverse_prob_fn(
        score, Y_earlier, Y_later, t1, step_size
    ).sum(-1)
    log_R = reverse_prob.log() - forward_prob.log()

    return log_R


def log_R(
    t1: Float[torch.Tensor, "batch"],
    t2: Float[torch.Tensor, "batch"],
    step_size: float,
    Y_path: List[Int[torch.Tensor, "batch datadim"]],
    score_path: List[Float[torch.Tensor, "batch datadim vocab"]],
    forward_prob_fn: Callable,
    reverse_prob_fn: Callable,
    device: torch.device,
    method: str = "score",
) -> Float[torch.Tensor, "batch"]:
    """
    Compute log_R using either scores or probabilities.
    """
    if method == "score":
        return _log_R_score(
            t1,
            t2,
            step_size,
            Y_path,
            score_path,
            forward_prob_fn,
            reverse_prob_fn,
            device,
        )
    elif method == "prob":
        return _log_R_prob(
            t1,
            t2,
            step_size,
            Y_path,
            score_path,
            forward_prob_fn,
            reverse_prob_fn,
            device,
        )
    else:
        raise ValueError(f"Invalid method: {method}")


class RNEWeightComputer(SMCWeightComputer):
    """SMC weight computation for Radon-Nikodym Estimator (RNE).

    This class implements the RNE weight update formula:

    log W_{k+1} = log W_k +
                  (1 - α) * log_R_uncond(Y_{[t_j, t_{j+1}]}) +
                  α * log_R_cond(Y_{[t_j, t_{j+1}]}) -
                  log_R_proposal(Y_{[t_j, t_{j+1}]})

    where log_R computes the Radon-Nikodym ratio for a given path and score.
    """

    def __init__(self, graph, noise, smc_temperature: float = 1.0):
        """Initialize the RNE weight computer.

        Args:
            graph: Rate matrix graph object
            noise: Noise schedule object
            smc_temperature: Temperature α for weighting unconditional vs conditional ratios
        """
        super().__init__(smc_temperature)
        from lib.sampler import get_analytic_prob_fn, get_euler_prob_fn

        self.forward_prob_fn, self.reverse_prob_fn = get_euler_prob_fn(graph, noise)
        # self.forward_prob_fn, self.reverse_prob_fn = get_analytic_prob_fn(graph, noise)

    def compute_weight_update(
        self,
        x_prev: Int[torch.Tensor, "batch datadim"],
        x_curr: Int[torch.Tensor, "batch datadim"],
        cond: torch.Tensor,
        proposal_log_prob: Float[torch.Tensor, "batch datadim"],
        uncond_log_prob: Float[torch.Tensor, "batch datadim"],
        cond_log_ratio: Float[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch"]:
        """Placeholder method for interface compatibility.

        RNE weight computation is handled by compute_rne_weight_update instead.
        """
        raise NotImplementedError("Use compute_rne_weight_update for RNE computation")

    def compute_rne_weight_update(
        self,
        x_earlier: Int[torch.Tensor, "batch datadim"],
        x_later: Int[torch.Tensor, "batch datadim"],
        t: Float[torch.Tensor, "batch"],
        dt: float,
        uncond_score: Float[torch.Tensor, "batch datadim vocab"],
        cond_score: Float[torch.Tensor, "batch datadim vocab"],
        proposal_score: Float[torch.Tensor, "batch datadim vocab"],
        device: torch.device,
        method: str = "score",
    ) -> Float[torch.Tensor, "batch"]:
        """Compute RNE weight update using log_R ratios.

        Args:
            x_earlier: Particle states at earlier time [batch, datadim]
            x_later: Particle states at later time [batch, datadim]
            t: Current time [batch]
            dt: Time step size
            uncond_score: Unconditional score [batch, datadim, vocab]
            cond_score: Conditional score [batch, datadim, vocab]
            proposal_score: Proposal (CFG) score [batch, datadim, vocab]
            device: Device for computation

        Returns:
            Log weight update [batch]
        """
        # # Verify all particles at same time
        # if not torch.allclose(t, t[0]):
        #     raise ValueError("All particles must be at same time")

        # Y_path in forward chronological time: [earlier_time, later_time]
        Y_path = [x_earlier, x_later]

        # Compute RNE ratios for each score type
        log_R_uncond: Float[torch.Tensor, "batch"] = log_R(
            t,
            t + dt,
            dt,
            Y_path,
            [uncond_score],
            self.forward_prob_fn,
            self.reverse_prob_fn,
            device,
            method,
        )

        log_R_cond: Float[torch.Tensor, "batch"] = log_R(
            t,
            t + dt,
            dt,
            Y_path,
            [cond_score],
            self.forward_prob_fn,
            self.reverse_prob_fn,
            device,
            method,
        )

        # print("Computing log_R_proposal")
        log_R_proposal: Float[torch.Tensor, "batch"] = log_R(
            t,
            t + dt,
            dt,
            Y_path,
            [proposal_score],
            self.forward_prob_fn,
            self.reverse_prob_fn,
            device,
            method,
        )

        # RNE weight update formula: (1-α)*log_R_uncond + α*log_R_cond - log_R_proposal
        log_weight_update: Float[torch.Tensor, "batch"] = (
            (1 - self.smc_temperature) * log_R_uncond
            + self.smc_temperature * log_R_cond
            - log_R_proposal
        )

        # DEBUG: Print RNE weight computation details
        # print_rne_weight_debug(
        #     log_R_uncond, log_R_cond, log_R_proposal, log_weight_update, self.smc_temperature
        # )

        return log_weight_update
