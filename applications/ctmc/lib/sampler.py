import abc
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch.nn import functional as F

import lib.models.utils as mutils
from lib import graph, noise
from lib.catsample import sample_categorical


@dataclass
class SMCOutput:
    """Output container for SMC sampling results.

    This dataclass provides a clean interface for returning multiple traces
    from SMC sampling, avoiding complex tuple unpacking.
    """

    particles: Int[torch.Tensor, "batch datadim"]  # Final sampled particles
    ess_trace: Optional[List[float]] = (
        None  # Effective Sample Size evolution over steps
    )
    weight_trace: Optional[Float[torch.Tensor, "batch steps"]] = (
        None  # Log weight evolution [batch, steps]
    )
    resampling_trace: Optional[Int[torch.Tensor, "steps"]] = (
        None  # Binary resampling events [steps]
    )
    particles_trace: Optional[Int[torch.Tensor, "batch steps datadim"]] = (
        None  # All particles over time [batch, steps, datadim]
    )


@dataclass
class PTOutput:
    """Output container for Parallel Tempering sampling results.

    This dataclass provides a clean interface for returning PT sampling results,
    following the same pattern as SMCOutput for consistency.
    """

    samples: List[Int[torch.Tensor, "datadim"]]  # Collected samples after burn-in
    final_path: Int[torch.Tensor, "time_steps datadim"]  # Final evolved path
    burn_in_samples: Optional[List[Int[torch.Tensor, "datadim"]]] = (
        None  # Collected samples during burn-in period (if keep_burn_in=True)
    )
    acceptance_rates: Optional[List[float]] = (
        None  # Metropolis acceptance rates per step
    )
    path_trace: Optional[List[Int[torch.Tensor, "time_steps datadim"]]] = (
        None  # Path evolution over time [steps, time_steps, datadim]
    )
    accept_count_per_timestep: Optional[torch.Tensor] = (
        None  # Count of accepts per time step [time_steps] - values range from 0 to num_steps
    )
    log_weights_reverse_per_timestep: Optional[Float[torch.Tensor, "time_steps"]] = (
        None  # Accumulated log weights for reverse time per timestep [time_steps]
    )
    log_weights_forward_per_timestep: Optional[Float[torch.Tensor, "time_steps"]] = (
        None  # Accumulated log weights for forward time per timestep [time_steps]
    )
    weights_count_per_timestep: Optional[torch.Tensor] = (
        None  # Count of weight updates per timestep [time_steps]
    )
    k_paths_storage: Optional[Int[torch.Tensor, "k time_steps datadim"]] = (
        None  # Circular buffer storing k paths at intervals [k, time_steps, datadim]
    )


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph: graph.Graph, noise: noise.Noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(
        self,
        score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

    @abc.abstractmethod
    def _sample_from_cfg_proposal(
        self,
        cfg_score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],  # Current particle states
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for token sequences
        t: Float[torch.Tensor, "batch"],  # Time steps for each particle
        step_size: float,  # Euler step size dt
        dtype: torch.dtype = torch.float32,
    ) -> tuple[
        Int[torch.Tensor, "batch datadim"],  # x_next: Next particle states
        Float[
            torch.Tensor, "batch datadim"
        ],  # proposal_log_prob: Log P(x_next|x) under CFG proposal
        Float[
            torch.Tensor, "batch datadim"
        ],  # uncond_log_prob: Log P(x_next|x) under unconditional
        Float[
            torch.Tensor, "batch datadim"
        ],  # cond_log_ratio: Log [P(ζ|x_next)/P(ζ|x)]
    ]:
        pass


def get_euler_prob_fn(graph: graph.Graph, noise: noise.Noise):
    def forward_euler_prob_fn(
        past_x: Int[torch.Tensor, "batch datadim"],  # Y_tn
        current_x: Int[torch.Tensor, "batch datadim"],  # Y_tn+1
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, "batch datadim"]:
        sigma, dsigma = noise(t)
        rate = graph.rate(past_x)
        probs: Float[torch.Tensor, "batch datadim vocab"] = (
            F.one_hot(current_x, num_classes=graph.dim).to(rate)
            + step_size * dsigma[..., None, None] * rate
        )
        probs = torch.gather(probs, -1, current_x[..., None]).squeeze(-1)
        probs = torch.nan_to_num(
            probs * (current_x == graph.dim - 1), nan=1.0, posinf=1.0, neginf=1.0
        )
        # clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-8)
        return probs

    def reverse_euler_prob_fn(
        score: Float[
            torch.Tensor, "batch datadim vocab"
        ],  # score = score_fn(current_x.to(dtype), sigma)
        past_x: Int[torch.Tensor, "batch datadim"],  # Y_tn
        current_x: Int[torch.Tensor, "batch datadim"],  # Y_tn+1
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, "batch datadim"]:
        sigma, dsigma = noise(t)
        reverse_rate_result = graph.reverse_rate(current_x, score)
        rev_rate = step_size * dsigma[..., None, None] * reverse_rate_result
        one_hot = F.one_hot(past_x, num_classes=graph.dim).to(rev_rate)
        probs: Float[torch.Tensor, "batch datadim vocab"] = one_hot + rev_rate
        gathered_probs = torch.gather(probs, -1, past_x[..., None]).squeeze(-1)
        gathered_probs = torch.nan_to_num(
            gathered_probs * (past_x == graph.dim - 1), nan=1.0, posinf=1.0, neginf=1.0
        )
        # clamp to avoid log(0)
        gathered_probs = torch.clamp(gathered_probs, min=1e-8)
        return gathered_probs

    return forward_euler_prob_fn, reverse_euler_prob_fn


def get_analytic_prob_fn(graph: graph.Graph, noise: noise.Noise):
    def forward_analytic_prob_fn(
        past_x: Int[torch.Tensor, "batch datadim"],  # Y_tn
        current_x: Int[torch.Tensor, "batch datadim"],  # Y_tn+1
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, "batch datadim"]:
        sigma, dsigma = noise(t)
        transition = graph.transition(past_x, dsigma[..., None])
        probs = torch.gather(transition, -1, current_x[..., None]).squeeze(-1)
        return probs

    def reverse_analytic_prob_fn(
        score: Float[
            torch.Tensor, "batch datadim vocab"
        ],  # score = score_fn(current_x.to(dtype), sigma)
        past_x: Int[torch.Tensor, "batch datadim"],  # Y_tn
        current_x: Int[torch.Tensor, "batch datadim"],  # Y_tn+1
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, "batch datadim"]:
        sigma, dsigma = noise(t)
        stag_score: Float[torch.Tensor, "batch datadim vocab"] = graph.staggered_score(
            score, dsigma[..., None]
        )
        probs: Float[torch.Tensor, "batch datadim vocab"] = (
            stag_score * graph.transp_transition(current_x, dsigma[..., None])
        )
        probs: Float[torch.Tensor, "batch datadim"] = torch.gather(
            probs, -1, past_x[..., None]
        ).squeeze(-1)

        return probs

    return forward_analytic_prob_fn, reverse_analytic_prob_fn


class EulerPredictor(Predictor):
    def update_fn(
        self,
        score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ) -> Int[torch.Tensor, "batch datadim"]:
        """
        Give a score, a current state x, a time t, and a step size, return the next state
        by sampling from p_{t + step_size|t}(x_{t + step_size} | x) =
        delta(x_{t + step_size}, x) + step_size * dsigma * score(x, sigma) * Q(x, x_{t + step_size})
        """
        sigma, dsigma = self.noise(
            t
        )  # Tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]
        score: Float[torch.Tensor, "batch datadim vocab"]
        score = score_fn(x.to(dtype), sigma)

        rev_rate = (
            step_size * dsigma[..., None, None] * self.graph.reverse_rate(x, score)
        )
        x, probs = self.graph.sample_rate(x, rev_rate)
        return x

    def _sample_from_cfg_proposal(
        self,
        cfg_score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],  # Current particle states
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for token sequences
        t: Float[torch.Tensor, "batch"],  # Time steps for each particle
        step_size: float,  # Euler step size dt
        dtype: torch.dtype = torch.float32,
    ) -> tuple[
        Int[torch.Tensor, "batch datadim"],  # x_next: Next particle states
        Float[
            torch.Tensor, "batch datadim"
        ],  # proposal score [p(.|ζ)/p(x|ζ)]^cfg_temperature * [p(.)/p(x)]^(1 - cfg_temperature)
        Float[torch.Tensor, "batch datadim"],  # uncond score [p(.)/p(x)]
        Float[torch.Tensor, "batch datadim"],  # cond_score [p(.|ζ)/p(x|ζ)]
    ]:
        # Get noise parameters at current time
        sigma: Float[torch.Tensor, "batch"]  # Cumulative noise σ̄(t): [batch]
        dsigma: Float[torch.Tensor, "batch"]  # Noise rate σ(t): [batch]
        sigma, dsigma = self.noise(t)

        # Get CFG scores with auxiliary outputs (unconditional and conditional scores)
        proposal_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # CFG combined score: [batch, datadim, vocab]
        uncond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Unconditional score: [batch, datadim, vocab]
        cond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Conditional score: [batch, datadim, vocab]
        proposal_score, (uncond_score, cond_score) = cfg_score_fn(
            x.to(dtype), cond, sigma, with_aux=True
        )

        # Use proposal score (CFG) to generate next state via Euler step
        proposal_rev_rate: Float[torch.Tensor, "batch datadim vocab"] = (
            step_size
            * dsigma[..., None, None]
            * self.graph.reverse_rate(x, proposal_score)
        )  # Proposal reverse rate: [batch, datadim, vocab]
        x_next: Int[torch.Tensor, "batch datadim"]
        x_next, _ = self.graph.sample_rate(
            x, proposal_rev_rate
        )  # Sample next state: [batch, datadim]

        return x_next, proposal_score, uncond_score, cond_score

    def smc_update_fn(
        self,
        cfg_score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],  # Current particle states
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for token sequences
        t: Float[torch.Tensor, "batch"],  # Time steps for each particle
        step_size: float,  # Euler step size dt
        proposal_strength: float = 1.0,  # CFG temperature for proposal distribution
        dtype: torch.dtype = torch.int64,
    ) -> tuple[
        Int[torch.Tensor, "batch datadim"],  # x_next: Next particle states
        Float[
            torch.Tensor, "batch datadim"
        ],  # proposal_log_prob: Log P(x_next|x) under CFG proposal
        Float[
            torch.Tensor, "batch datadim"
        ],  # uncond_log_prob: Log P(x_next|x) under unconditional
        Float[
            torch.Tensor, "batch datadim"
        ],  # cond_log_ratio: Log [P(ζ|x_next)/P(ζ|x)]
    ]:
        """SMC update function that returns both next state and auxiliary information.

        This function performs one Euler step using a CFG proposal and returns
        the log probabilities needed for SMC weight computation:
        1. Proposal log probabilities (CFG with proposal_strength)
        2. Unconditional log probabilities (base process)
        3. Conditional likelihood ratios for weight computation

        Args:
            cfg_score_fn: CFG score function that takes (x, cond, sigma, with_aux)
            x: Current particle states [batch, datadim]
            cond: Conditioning variable (e.g., class labels)
            t: Current time [batch]
            step_size: Euler step size
            proposal_strength: CFG temperature for proposal distribution
            dtype: Data type for computation

        Returns:
            Tuple of:
            - x_next: Next particle states [batch, datadim]
            - proposal_log_prob: Log transition probabilities under proposal [batch, datadim]
            - uncond_log_prob: Log transition probabilities under unconditional [batch, datadim]
            - cond_log_ratio: Conditional likelihood ratios [batch, datadim]
        """
        # from lib.smc_weights import (
        #     compute_transition_log_probs,
        #     extract_cond_log_ratios,
        # )

        _, reverse_prob_fn = get_euler_prob_fn(self.graph, self.noise)

        # Get noise parameters at current time
        sigma: Float[torch.Tensor, "batch"]  # Cumulative noise σ̄(t): [batch]
        dsigma: Float[torch.Tensor, "batch"]  # Noise rate σ(t): [batch]
        sigma, dsigma = self.noise(t)

        # Get CFG scores with auxiliary outputs (unconditional and conditional scores)
        proposal_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # CFG combined score: [batch, datadim, vocab]
        uncond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Unconditional score: [batch, datadim, vocab]
        cond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Conditional score: [batch, datadim, vocab]
        proposal_score, (uncond_score, cond_score) = cfg_score_fn(
            x.to(dtype), cond, sigma, with_aux=True
        )

        # Use proposal score (CFG) to generate next state via Euler step
        proposal_rev_rate: Float[torch.Tensor, "batch datadim vocab"] = (
            step_size
            * dsigma[..., None, None]
            * self.graph.reverse_rate(x, proposal_score)
        )  # Proposal reverse rate: [batch, datadim, vocab]
        x_next: Int[torch.Tensor, "batch datadim"]
        x_next, _ = self.graph.sample_rate(
            x, proposal_rev_rate
        )  # Sample next state: [batch, datadim]

        proposal_log_prob: Float[torch.Tensor, "batch datadim"] = reverse_prob_fn(
            proposal_score, x_next, x, t, step_size
        ).log()

        # # Compute unconditional reverse rate for weight computation
        # uncond_rev_rate: Float[torch.Tensor, "batch datadim vocab"] = (
        #     step_size
        #     * dsigma[..., None, None]
        #     * self.graph.reverse_rate(x, uncond_score)
        # )  # Unconditional reverse rate: [batch, datadim, vocab]

        uncond_log_prob: Float[torch.Tensor, "batch datadim"] = reverse_prob_fn(
            uncond_score, x_next, x, t, step_size
        ).log()

        cond_log_prob: Float[torch.Tensor, "batch datadim"] = reverse_prob_fn(
            cond_score, x_next, x, t, step_size
        ).log()

        # Compute log transition probabilities using CTMC formula: P(x_next|x) = δ + R*dt
        # proposal_log_prob: Float[torch.Tensor, "batch datadim"] = (
        #     compute_transition_log_probs(self.graph, x, x_next, proposal_rev_rate)
        # )  # Log P(x_next|x) under proposal: [batch, datadim]
        # uncond_log_prob: Float[torch.Tensor, "batch datadim"] = (
        #     compute_transition_log_probs(self.graph, x, x_next, uncond_rev_rate)
        # )  # Log P(x_next|x) under unconditional: [batch, datadim]

        # # Extract conditional likelihood ratios using Bayes' rule
        # # Log [P(ζ|x_next)/P(ζ|x)] = Log [P(x_next|ζ)/P(x|ζ)] + Log [P(x)/P(x_next)]
        # cond_log_ratio: Float[torch.Tensor, "batch datadim"] = extract_cond_log_ratios(
        #     uncond_score, cond_score, x_next
        # )  # Log conditional likelihood ratio: [batch, datadim]

        twist = cond_log_prob - uncond_log_prob

        return x_next, proposal_log_prob, uncond_log_prob, twist


class AnalyticPredictor(Predictor):
    def update_fn(
        self, score_fn, x, t, step_size
    ) -> Int[torch.Tensor, "batch datadim"]:
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score: Float[torch.Tensor, "batch datadim vocab"] = score_fn(x, curr_sigma)
        stag_score: Float[torch.Tensor, "batch datadim vocab"] = (
            self.graph.staggered_score(score, dsigma[..., None])
        )
        probs: Float[torch.Tensor, "batch datadim vocab"] = (
            stag_score * self.graph.transp_transition(x, dsigma[..., None])
        )
        return sample_categorical(probs)

    def smc_update_fn(
        self,
        cfg_score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],  # Current particle states
        cond: torch.Tensor,  # Conditioning: [batch] for class labels OR [batch, datadim] for token sequences
        t: Float[torch.Tensor, "batch"],  # Time steps for each particle
        step_size: float,  # Step size dt
        proposal_strength: float = 1.0,  # CFG temperature for proposal distribution
        dtype: torch.dtype = torch.float32,
    ) -> tuple[
        Int[torch.Tensor, "batch datadim"],  # x_next: Next particle states
        Float[
            torch.Tensor, "batch datadim"
        ],  # proposal_log_prob: Log P(x_next|x) under CFG proposal
        Float[
            torch.Tensor, "batch datadim"
        ],  # uncond_log_prob: Log P(x_next|x) under unconditional
        Float[
            torch.Tensor, "batch datadim"
        ],  # cond_log_ratio: Log [P(ζ|x_next)/P(ζ|x)]
    ]:
        """SMC update function using analytic transition matrices.

        This function performs one analytic step using CFG proposal and returns
        the log probabilities needed for SMC weight computation. Uses exact
        discrete transition matrices instead of Euler approximation.

        Args:
            cfg_score_fn: CFG score function that takes (x, cond, sigma, with_aux)
            x: Current particle states [batch, datadim]
            cond: Conditioning variable (e.g., class labels)
            t: Current time [batch]
            step_size: Step size for noise schedule
            proposal_strength: CFG temperature for proposal distribution
            dtype: Data type for computation

        Returns:
            Tuple of:
            - x_next: Next particle states [batch, datadim]
            - proposal_log_prob: Log transition probabilities under proposal [batch, datadim]
            - uncond_log_prob: Log transition probabilities under unconditional [batch, datadim]
            - cond_log_ratio: Conditional likelihood ratios [batch, datadim]
        """
        # Compute noise schedule parameters
        curr_sigma = self.noise(t)[0]  # Current cumulative noise: [batch]
        next_sigma = self.noise(t - step_size)[0]  # Next cumulative noise: [batch]
        dsigma = curr_sigma - next_sigma  # Noise step: [batch]

        # Get CFG scores with auxiliary outputs
        proposal_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # CFG combined score: [batch, datadim, vocab]
        uncond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Unconditional score: [batch, datadim, vocab]
        cond_score: Float[
            torch.Tensor, "batch datadim vocab"
        ]  # Conditional score: [batch, datadim, vocab]
        proposal_score, (uncond_score, cond_score) = cfg_score_fn(
            x.to(dtype), cond, curr_sigma, with_aux=True
        )

        # Compute transition probabilities using proposal (CFG) score
        proposal_stag_score: Float[torch.Tensor, "batch datadim vocab"] = (
            self.graph.staggered_score(proposal_score, dsigma[..., None])
        )
        proposal_probs: Float[torch.Tensor, "batch datadim vocab"] = (
            proposal_stag_score * self.graph.transp_transition(x, dsigma[..., None])
        )

        # Sample next state using proposal distribution
        x_next: Int[torch.Tensor, "batch datadim"] = sample_categorical(proposal_probs)

        # Extract log probabilities for the actual sampled transition: x -> x_next
        proposal_log_prob: Float[torch.Tensor, "batch datadim"] = torch.gather(
            proposal_probs.log(), -1, x_next[..., None]
        ).squeeze(-1)

        # Compute unconditional transition probabilities for the same transition
        uncond_stag_score: Float[torch.Tensor, "batch datadim vocab"] = (
            self.graph.staggered_score(uncond_score, dsigma[..., None])
        )
        uncond_probs: Float[torch.Tensor, "batch datadim vocab"] = (
            uncond_stag_score * self.graph.transp_transition(x, dsigma[..., None])
        )
        uncond_log_prob: Float[torch.Tensor, "batch datadim"] = torch.gather(
            uncond_probs.log(), -1, x_next[..., None]
        ).squeeze(-1)

        # Apply absorbing graph masking (following colleague's logic)
        # Only consider transitions from absorbing state (x == self.graph.dim - 1)
        if (
            self.graph.absorb
        ):  # TODO: Check how this would work for non absorbing graphs
            absorbing_mask = (x == self.graph.dim - 1).float()
            proposal_log_prob = proposal_log_prob * absorbing_mask
            uncond_log_prob = uncond_log_prob * absorbing_mask

            # # Handle NaN/Inf values that may arise from log(0)
            # proposal_log_prob = torch.nan_to_num(
            #     proposal_log_prob, nan=0.0, posinf=0.0, neginf=0.0
            # )  # TODO: Mode collapse can come from here.
            # uncond_log_prob = torch.nan_to_num(
            #     uncond_log_prob, nan=0.0, posinf=0.0, neginf=0.0
            # )

        # # Extract conditional likelihood ratios using Bayes' rule
        # # This uses your existing CFG auxiliary output format: (uncond_score, cond_score)
        # cond_log_ratio: Float[torch.Tensor, "batch datadim"] = extract_cond_log_ratios(
        #     uncond_score, cond_score, x_next
        # )

        twist_log: Float[torch.Tensor, "batch datadim vocab"] = (
            cond_score.log() - uncond_score.log()
        )
        cond_log_ratio = twist_log.gather(-1, x_next[..., None]).squeeze(-1)
        # Apply masking to conditional likelihood ratio for actual transitions only
        if self.graph.absorb:
            # cond_log_ratio = torch.nan_to_num(
            #     cond_log_ratio * (x != x_next).float(), nan=0.0, posinf=0.0, neginf=0.0
            # )
            cond_log_ratio = cond_log_ratio * (x != x_next).float()

            # transition_mask = (x != x_next).float()
            # cond_log_ratio = cond_log_ratio * transition_mask
            # cond_log_ratio = torch.nan_to_num(
            #     cond_log_ratio, nan=0.0, posinf=0.0, neginf=0.0
            # )

        return x_next, proposal_log_prob, uncond_log_prob, cond_log_ratio


class NonePredictor(Predictor):
    def update_fn(
        self,
        score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],
        t: Float[torch.Tensor, "batch"],
        step_size: float,
        dtype: torch.dtype = torch.float32,
    ):
        return x


class Denoiser:
    def __init__(self, graph: graph.Graph, noise: noise.Noise):
        self.graph = graph
        self.noise = noise

    def update_fn(
        self,
        score_fn: Callable,
        x: Int[torch.Tensor, "batch datadim"],
        t: Float[torch.Tensor, "batch"],
        sigma: Float[torch.Tensor, "batch"] = None,
        return_probs: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> Int[torch.Tensor, "batch datadim"]:
        """
        Performs denoising by computing the posterior distribution over clean states.

        This function computes p(x_0 | x_t) by:
        1. Getting the score from the model: score(x_t, sigma_t)
        2. Computing staggered score: p_{t-dt}(z) / p_t(x) ≈ e^{-dt*E} * score
        3. Computing transition probabilities: stag_score * transp_transition(x_t, sigma_t)
        4. Sampling from the resulting distribution over clean states

        For absorbing graphs, truncates the absorbing state from final probabilities.

        Args:
            score_fn: Model score function that takes (x, sigma) and returns scores
            x: Current noisy state tensor of shape [batch, datadim]
            t: Time step tensor of shape [batch]
            sigma: Noise level tensor of shape [batch] (It's either sigma or t, but it can not be both. We assume t is given and sigma is computed from t)
            return_probs: If True, returns argmax instead of sampling
            dtype: Data type for computation

        Returns:
            Denoised state tensor of shape [batch, datadim]
        """
        if sigma is None and t is None:
            raise ValueError("Either sigma or t must be provided")

        if sigma is not None and t is not None:
            raise ValueError("Either sigma or t must be provided, but not both")

        if t is not None:
            sigma, _ = self.noise(t)
        else:
            sigma = sigma

        score: Float[torch.Tensor, "batch datadim vocab"] = score_fn(x.to(dtype), sigma)
        stag_score: Float[torch.Tensor, "batch datadim vocab"] = (
            self.graph.staggered_score(score, sigma[..., None])
        )
        probs: Float[torch.Tensor, "batch datadim vocab"] = (
            stag_score * self.graph.transp_transition(x.long(), sigma[..., None])
        )
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        if return_probs:
            return probs.argmax(dim=-1)

        return sample_categorical(probs)


def _execute_sampling_loop(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool,
    eps: float,
    device: torch.device,
    proj_fun: Callable,
    sampling_schedule: str,
    return_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Execute the standard SEDD sampling loop.

    This function contains ALL the shared logic between unconditional,
    CFG, and DEFT samplers. The only difference between samplers is
    how they create the score_fn parameter.

    Args:
        score_fn: Pre-configured score function that takes (x, t) and returns scores
        graph: Rate matrix graph (Uniform or Absorbing)
        noise: Noise schedule (LogLinear, Geometric, etc.)
        batch_dims: Dimensions for initial particle sampling
        predictor: Predictor algorithm (EulerPredictor, AnalyticPredictor, etc.)
        steps: Number of diffusion steps
        denoise: Whether to perform final denoising step
        eps: Final time (close to 0)
        device: Device for computation
        proj_fun: Projection function (identity by default)
        sampling_schedule: Time discretization schedule ("linear" or "cosine")
        return_path: Whether to return the full sampling trajectory

    Returns:
        If return_path=False: Tuple of (final_samples, initial_samples)
        If return_path=True: Tuple of (final_samples, initial_samples, path_tensor)
        where path_tensor has shape [steps+1, batch, datadim]
    """
    assert callable(score_fn), "score_fn must be callable"
    assert steps > 0, "steps must be positive"

    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    # Initialize sampling state
    x = graph.sample_limit(*batch_dims).to(device)
    initial_x = x

    # Initialize path storage if requested
    path_tensor = None
    if return_path:
        path_tensor = torch.zeros(steps + 1, *x.shape, device=device, dtype=x.dtype)
        path_tensor[0] = x.clone()  # Store initial state

    # Create time discretization
    timesteps = sampling_schedule_grid(sampling_schedule, steps, eps, device)
    dt = (1 - eps) / steps

    try:
        # Main sampling loop
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], device=device)
            x = projector(x)
            x = predictor.update_fn(score_fn, x, t, dt, dtype=x.dtype)

            # Store intermediate state if path requested
            if return_path:
                path_tensor[i + 1] = x.clone()

        # Optional denoising step
        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], device=device)
            x = denoiser.update_fn(score_fn, x, t, dtype=x.dtype)

            # Update final path state if needed
            if return_path:
                path_tensor[-1] = x.clone()

        if return_path:
            return x, initial_x, path_tensor
        else:
            return x, initial_x

    except Exception as e:
        raise RuntimeError(
            f"Sampling failed at step {i if 'i' in locals() else 'initialization'}/{steps}: {e}"
        ) from e


def _create_unconditional_score_fn(model):
    """Create simple unconditional score function.

    Args:
        model: The unconditional diffusion model

    Returns:
        Score function that takes (x, t) and returns scores
    """
    return mutils.get_score_fn(model, train=False, sampling=True)


def _create_cfg_score_fn(uncond_model, cond_model, cond, cfg_temperature):
    """Create CFG score function with conditioning.

    Args:
        uncond_model: Unconditional diffusion model
        cond_model: Conditional diffusion model
        cond: Conditioning variable (e.g., class labels)
        cfg_temperature: Classifier-free guidance strength

    Returns:
        Score function that takes (x, t) and returns CFG-guided scores
    """
    sampling_score_fn = mutils.get_cfg_score_fn(
        uncond_model, cond_model, cfg_temperature, train=False, sampling=True
    )
    return lambda x_in, t_in: sampling_score_fn(x_in, cond, t_in)


def _create_cfg_finetune_score_fn(uncond_model, cond_model, cond, cfg_temperature):
    """Create CFG score function with conditioning for FinetuneSEDD models.
    Args:
        uncond_model: Unconditional diffusion model (base SEDD)
        cond_model: Conditional diffusion model (FinetuneSEDD)
        cond: Conditioning variable (e.g., class labels)
        cfg_temperature: Classifier-free guidance strength
    Returns:
        Score function that takes (x, t) and returns CFG-guided scores
    """
    sampling_score_fn = mutils.get_cfg_score_finetune_fn(
        uncond_model, cond_model, cfg_temperature, train=False, sampling=True
    )
    return lambda x_in, t_in: sampling_score_fn(x_in, cond, t_in)


def _create_cfg_discdiff_score_fn(uncond_model, cond_model, cond, cfg_temperature):
    """Create CFG score function with conditioning for the DiscDiff model.

    Args:
        uncond_model: Unconditional diffusion model
        cond_model: Conditional diffusion model
        cond: Conditioning variable (e.g., class labels)
        cfg_temperature: Classifier-free guidance strength

    Returns:
        Score function that takes (x, t) and returns CFG-guided scores
    """
    sampling_score_fn = mutils.get_score_discdiff_fn(
        uncond_model, cond_model, cfg_temperature, train=False, sampling=True
    )
    return lambda x_in, t_in: sampling_score_fn(x_in, cond, t_in)


def _create_deft_score_fn(
    uncond_model, cond_model, cond, deft_temperature, graph, noise
):
    """Create DEFT score function with denoising guidance.

    Args:
        uncond_model: Unconditional diffusion model
        cond_model: Conditional diffusion model
        cond: Conditioning variable (e.g., class labels)
        deft_temperature: DEFT guidance strength
        graph: Rate matrix graph (needed for denoiser)
        noise: Noise schedule (needed for denoiser)

    Returns:
        Score function that takes (x, t) and returns DEFT-guided scores
    """
    score_fn = mutils.get_score_fn(uncond_model, train=False, sampling=True)
    denoiser = Denoiser(graph, noise)
    denoiser_fn = lambda x, sigma: denoiser.update_fn(
        score_fn, x, t=None, sigma=sigma, dtype=x.dtype
    )
    sampling_score_fn = mutils.get_deft_score_fn(
        uncond_model,
        cond_model,
        denoiser_fn,
        deft_temperature,
        train=False,
        sampling=True,
    )
    return lambda x_in, t_in: sampling_score_fn(x_in, cond, t_in)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    sampling_fn = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
    )

    return sampling_fn


def sampling_schedule_grid(
    schedule: str, steps: int, eps: float, device: torch.device
) -> Float[torch.Tensor, "steps"]:
    if schedule == "linear":
        return torch.linspace(1, eps, steps + 1, device=device)
    elif schedule == "cosine":  # Jiaxin's schedule Eq 6
        return torch.cos(
            torch.pi / 2 * (1 - torch.linspace(1, eps, steps + 1, device=device))
        )
    else:
        raise ValueError(f"Invalid schedule: {schedule}")


def get_pc_sampler(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool = True,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable = lambda x: x,
    sampling_schedule: str = "linear",
    return_path: bool = False,
):
    """Create unconditional sampler - uses basic score function."""

    @torch.no_grad()
    def pc_sampler(model):
        score_fn = _create_unconditional_score_fn(model)
        return _execute_sampling_loop(
            score_fn=score_fn,
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun,
            sampling_schedule=sampling_schedule,
            return_path=return_path,
        )

    return pc_sampler


def get_pc_sampler_cfg(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool = True,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable = lambda x: x,
    sampling_schedule: str = "linear",
    return_path: bool = False,
):
    """Create CFG sampler - uses classifier-free guidance."""

    @torch.no_grad()
    def sampler(uncond_model, cond_model, cond, cfg_temperature: float = 1.0):
        score_fn = _create_cfg_score_fn(uncond_model, cond_model, cond, cfg_temperature)
        return _execute_sampling_loop(
            score_fn=score_fn,
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun,
            sampling_schedule=sampling_schedule,
            return_path=return_path,
        )

    return sampler


def get_pc_sampler_cfg_finetune(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool = True,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable = lambda x: x,
    sampling_schedule: str = "linear",
    return_path: bool = False,
):
    """Create CFG sampler for FinetuneSEDD models - uses classifier-free guidance."""

    @torch.no_grad()
    def sampler(uncond_model, cond_model, cond, cfg_temperature: float = 1.0):
        score_fn = _create_cfg_finetune_score_fn(
            uncond_model, cond_model, cond, cfg_temperature
        )
        return _execute_sampling_loop(
            score_fn=score_fn,
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun,
            sampling_schedule=sampling_schedule,
            return_path=return_path,
        )

    return sampler


def get_pc_sampler_cfg_discdiff(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool = True,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable = lambda x: x,
    sampling_schedule: str = "linear",
    return_path: bool = False,
):
    """Create CFG sampler - uses classifier-free guidance."""

    @torch.no_grad()
    def sampler(uncond_model, cond_model, cond, cfg_temperature: float = 1.0):
        score_fn = _create_cfg_discdiff_score_fn(
            uncond_model, cond_model, cond, cfg_temperature
        )
        return _execute_sampling_loop(
            score_fn=score_fn,
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun,
            sampling_schedule=sampling_schedule,
            return_path=return_path,
        )

    return sampler


def get_pc_sampler_deft(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    denoise: bool = True,
    eps: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable = lambda x: x,
    sampling_schedule: str = "linear",
    return_path: bool = False,
):
    """Create DEFT sampler - uses denoising guidance."""

    @torch.no_grad()
    def sampler(uncond_model, cond_model, cond, deft_temperature: float = 1.0):
        score_fn = _create_deft_score_fn(
            uncond_model, cond_model, cond, deft_temperature, graph, noise
        )
        return _execute_sampling_loop(
            score_fn=score_fn,
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun,
            sampling_schedule=sampling_schedule,
            return_path=return_path,
        )

    return sampler


def _create_trace_containers(
    traces: List[str], particle_shape: Tuple[int, ...], steps: int, device: torch.device
) -> dict:
    """Create trace storage containers based on requested traces.

    Args:
        traces: List of trace types to collect
        particle_shape: Shape of particle tensor [batch, datadim, ...]
        steps: Number of sampling steps
        device: Device for tensor allocation

    Returns:
        Dictionary of trace containers
    """
    containers = {}

    if "ess" in traces:
        containers["ess_trace"] = []

    if "weight" in traces:
        containers["weight_trace"] = torch.zeros(
            particle_shape[0], steps, device=device
        )

    if "resampling" in traces:
        containers["resampling_trace"] = torch.zeros(
            steps, dtype=torch.int32, device=device
        )

    if "particles" in traces:
        containers["particles_trace"] = torch.zeros(
            particle_shape[0],
            steps,
            *particle_shape[1:],
            dtype=torch.long,
            device=device,
        )

    return containers


def _initialize_smc_state(
    graph: graph.Graph,
    batch_dims: Tuple[int, ...],
    device: torch.device,
    traces: List[str],
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Initialize particles, weights, and trace containers for SMC sampling.

    Args:
        graph: Rate matrix graph for particle initialization
        batch_dims: Dimensions for particle sampling
        device: Device for tensor allocation
        traces: List of trace types to collect
        steps: Number of sampling steps

    Returns:
        Tuple of (initial_particles, initial_log_weights, trace_containers)
    """
    # Initialize particles from limit distribution
    x = graph.sample_limit(*batch_dims).to(device)

    # Initialize uniform log weights (all particles equally likely)
    log_weights = torch.zeros(x.shape[0], device=device)

    # Create trace containers
    trace_containers = _create_trace_containers(traces, x.shape, steps, device)

    return x, log_weights, trace_containers


def _rne_step(
    predictor: Predictor,
    cfg_score_fn: Callable,
    x: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    proposal_strength: float,
    weight_computer,
    device: torch.device,
    method: str = "score",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one RNE step with weight update.

    Args:
        predictor: Predictor with _sample_from_cfg_proposal support
        cfg_score_fn: CFG score function
        x: Current particle states
        cond: Conditioning variable
        t: Current time
        dt: Time step size
        proposal_strength: CFG temperature for proposal
        weight_computer: RNE weight computation object
        device: Device for computation

    Returns:
        Tuple of (next_particles, log_weight_update)
    """
    # Get next particles and score tensors using existing method
    x_next, proposal_score, uncond_score, cond_score = (
        predictor._sample_from_cfg_proposal(cfg_score_fn, x, cond, t, dt, dtype=x.dtype)
    )

    # Compute RNE weight update
    # Pass states in forward chronological time: [earlier, later] = [x_next, x]
    log_weight_update = weight_computer.compute_rne_weight_update(
        x_next, x, t, dt, uncond_score, cond_score, proposal_score, device, method
    )

    return x_next, log_weight_update


def _smc_step(
    predictor: Predictor,
    cfg_score_fn: Callable,
    x: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    proposal_strength: float,
    weight_computer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform one SMC step with weight update.

    Args:
        predictor: Predictor with smc_update_fn support
        cfg_score_fn: CFG score function
        x: Current particle states
        cond: Conditioning variable
        t: Current time
        dt: Time step size
        proposal_strength: CFG temperature for proposal
        weight_computer: SMC weight computation object

    Returns:
        Tuple of (next_particles, log_weight_update)
    """
    # Get next particles and auxiliary information for weight computation
    x_next, proposal_log_prob, uncond_log_prob, cond_log_ratio = (
        predictor.smc_update_fn(
            cfg_score_fn, x, cond, t, dt, proposal_strength, dtype=x.dtype
        )
    )

    # Compute importance weight update
    log_weight_update = weight_computer.compute_weight_update(
        x, x_next, cond, proposal_log_prob, uncond_log_prob, cond_log_ratio
    )

    return x_next, log_weight_update


def _update_traces(
    trace_containers: dict, x: torch.Tensor, log_weights: torch.Tensor, step: int
) -> dict:
    """Update all requested traces with current state.

    Args:
        trace_containers: Dictionary of trace storage containers
        x: Current particle states
        log_weights: Current log weights
        step: Current step index

    Returns:
        Dictionary of trace containers
    """
    if "particles_trace" in trace_containers:
        trace_containers["particles_trace"][:, step, :] = x

    if "weight_trace" in trace_containers:
        trace_containers["weight_trace"][:, step] = log_weights

    return trace_containers


def _handle_resampling(
    x: torch.Tensor,
    log_weights: torch.Tensor,
    step: int,
    steps: int,
    resampling_params: dict,
    trace_containers: dict,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Handle adaptive resampling logic and trace updates.

    Args:
        x: Current particle states
        log_weights: Current log weights
        step: Current step index
        steps: Total number of steps
        resampling_params: Dictionary of resampling configuration
        trace_containers: Dictionary of trace storage containers

    Returns:
        Tuple of (resampled_particles, resampled_weights, ess_value)
    """
    from lib.resampling import compute_ess_from_log_weights, should_resample

    # Compute effective sample size
    ess = compute_ess_from_log_weights(log_weights)

    # Update ESS trace
    if "ess_trace" in trace_containers:
        trace_containers["ess_trace"].append(ess.item())

    # Check if resampling is needed
    min_step = int(resampling_params["sampling_threshold"] * steps)
    if step >= min_step and should_resample(
        log_weights, resampling_params["ess_threshold"]
    ):
        # Record resampling event
        if "resampling_trace" in trace_containers:
            trace_containers["resampling_trace"][step] = 1

        # Perform resampling
        resampling_fn = resampling_params["resampling_fn"]
        if resampling_params["method"] == "partial":
            x, log_weights = resampling_fn(
                x, log_weights, resampling_params["fraction"]
            )
        else:
            x, log_weights = resampling_fn(x, log_weights)

        # Update traces
        trace_containers = _update_traces(trace_containers, x, log_weights, step)

    return x, log_weights, ess.item(), trace_containers


def _execute_smc_loop(
    predictor: Predictor,
    cfg_score_fn: Callable,
    x: torch.Tensor,
    log_weights: torch.Tensor,
    cond: torch.Tensor,
    timesteps: torch.Tensor,
    dt: float,
    proposal_strength: float,
    weight_computer,
    resampling_params: dict,
    trace_containers: dict,
    projector: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute the main SMC sampling loop.

    Args:
        predictor: Predictor algorithm
        cfg_score_fn: CFG score function
        x: Initial particle states
        log_weights: Initial log weights
        cond: Conditioning variable
        timesteps: Time discretization grid
        dt: Time step size
        proposal_strength: CFG temperature for proposal
        weight_computer: SMC weight computation object
        resampling_params: Resampling configuration
        trace_containers: Trace storage containers
        projector: Projection function

    Returns:
        Tuple of (final_particles, final_log_weights)
    """
    steps = len(timesteps) - 1

    for i in range(steps):
        # Prepare for this step
        t = timesteps[i] * torch.ones(x.shape[0], device=x.device)
        x = projector(x)

        # Perform SMC step
        x_next, log_weight_update = _smc_step(
            predictor, cfg_score_fn, x, cond, t, dt, proposal_strength, weight_computer
        )
        log_weights += log_weight_update

        # Update traces
        trace_containers = _update_traces(trace_containers, x_next, log_weights, i)

        # Handle adaptive resampling
        x_next, log_weights, ess, trace_containers = _handle_resampling(
            x_next, log_weights, i, steps, resampling_params, trace_containers
        )

        x = x_next

    return x, log_weights, trace_containers


def _handle_denoising(
    x: torch.Tensor,
    log_weights: torch.Tensor,
    cfg_score_fn: Callable,
    cond: torch.Tensor,
    timesteps: torch.Tensor,
    denoiser,
    projector: Callable,
    resampling_params: dict,
    trace_containers: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handle final denoising step with optional resampling.

    Args:
        x: Pre-denoising particle states
        log_weights: Current log weights
        cfg_score_fn: CFG score function
        cond: Conditioning variable
        timesteps: Time discretization grid
        denoiser: Denoising object
        projector: Projection function
        resampling_params: Resampling configuration
        trace_containers: Trace storage containers

    Returns:
        Tuple of (denoised_particles, final_log_weights)
    """
    from lib.resampling import compute_ess_from_log_weights, should_resample

    # Apply final projection
    x = projector(x)

    # Final time ≈ eps
    t = timesteps[-1] * torch.ones(x.shape[0], device=x.device)

    # Denoise using Tweedie's formula
    x = denoiser.update_fn(
        lambda x_in, t_in: cfg_score_fn(x_in, cond, t_in), x, t, dtype=x.dtype
    )

    # Final ESS check
    ess = compute_ess_from_log_weights(log_weights)
    if "ess_trace" in trace_containers:
        trace_containers["ess_trace"].append(ess.item())

    # Optional final resampling
    if should_resample(log_weights, resampling_params["ess_threshold"]):
        resampling_fn = resampling_params["resampling_fn"]
        if resampling_params["method"] == "partial":
            x, log_weights = resampling_fn(
                x, log_weights, resampling_params["fraction"]
            )
        else:
            x, log_weights = resampling_fn(x, log_weights)

    # Update final particle trace if requested
    if "particles_trace" in trace_containers:
        # Use the last step index for final particles
        final_step = trace_containers["particles_trace"].shape[1] - 1
        trace_containers["particles_trace"][:, final_step, :] = x

    return x, log_weights, trace_containers


def get_cfg_smc_sampler(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    proposal_strength: float = 1.0,
    smc_temperature: float = 1.0,
    denoise: bool = True,
    eps: float = 1e-5,
    ess_threshold: float = 0.5,
    resample_fraction: float = 0.2,
    sampling_threshold: float = 0.1,
    resampling_method: str = "partial",
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable[
        [Int[torch.Tensor, "batch datadim"]], Int[torch.Tensor, "batch datadim"]
    ] = lambda x: x,
    sampling_schedule: str = "linear",
    score_fn_type: str = "standard",
) -> Callable:
    """Create an SMC sampler for unbiased CFG sampling.

    This sampler implements Sequential Monte Carlo to sample from the true
    tempered distribution p_t^α(x_t) ∝ p_t(x_t) p_t(ζ | x_t)^α, avoiding
    the bias introduced by standard classifier-free guidance.

    Args:
        graph: Rate matrix graph (Uniform or Absorbing)
        noise: Noise schedule (LogLinear, Geometric, etc.)
        batch_dims: Dimensions for initial particle sampling
        predictor: Predictor algorithm (should be EulerPredictor for SMC support)
        steps: Number of diffusion steps
        proposal_strength: CFG temperature for proposal distribution q
        smc_temperature: Temperature α for likelihood ratio weighting
        denoise: Whether to perform final denoising step
        eps: Final time (close to 0)
        ess_threshold: ESS threshold for triggering resampling (fraction of particle count)
        resample_fraction: Fraction of particles to resample when triggered
        sampling_threshold: Don't resample before this fraction of steps
        resampling_method: Resampling strategy ("partial", "multinomial", "stratified")
        device: Device for computation
        proj_fun: Projection function (identity by default)
        sampling_schedule: Time discretization schedule ("linear" or "cosine")
        score_fn_type: Score function type ("standard", "discdiff", or "finetune")

    Returns:
        SMC sampler function that takes (uncond_model, cond_model, cond)
    """
    from lib.resampling import get_resampling_fn
    from lib.smc_weights import CFGWeightComputer

    # Validation
    if not isinstance(predictor, (EulerPredictor, AnalyticPredictor)):
        raise ValueError(
            "SMC sampler requires EulerPredictor or AnalyticPredictor for auxiliary outputs"
        )

    # Setup shared components
    projector = proj_fun
    denoiser = Denoiser(graph, noise)
    weight_computer = CFGWeightComputer(smc_temperature=smc_temperature)
    resampling_fn = get_resampling_fn(resampling_method)

    # Package resampling parameters
    resampling_params = {
        "method": resampling_method,
        "ess_threshold": ess_threshold,
        "fraction": resample_fraction,
        "sampling_threshold": sampling_threshold,
        "resampling_fn": resampling_fn,
    }

    @torch.no_grad()
    def smc_sampler(
        uncond_model: torch.nn.Module,
        cond_model: torch.nn.Module,
        cond: torch.Tensor,
        traces: List[str] = None,
    ) -> SMCOutput:
        """SMC sampler for unbiased CFG.

        Args:
            uncond_model: Unconditional diffusion model
            cond_model: Conditional diffusion model
            cond: Conditioning variable (e.g., class labels [batch] or token sequences [batch, datadim])
            traces: List of traces to collect. Options: ["ess", "weight", "resampling", "particles"]

        Returns:
            SMCOutput dataclass containing final particles and requested traces
        """
        # 1. Initialize SMC state
        x, log_weights, trace_containers = _initialize_smc_state(
            graph, batch_dims, device, traces or [], steps
        )

        # 2. Create CFG score function based on type
        if score_fn_type == "discdiff":
            cfg_score_fn = mutils.get_score_discdiff_fn(
                uncond_model, cond_model, proposal_strength, train=False, sampling=True
            )
        elif score_fn_type == "finetune":
            cfg_score_fn = mutils.get_cfg_score_finetune_fn(
                uncond_model, cond_model, proposal_strength, train=False, sampling=True
            )
        else:  # standard
            cfg_score_fn = mutils.get_cfg_score_fn(
                uncond_model, cond_model, proposal_strength, train=False, sampling=True
            )

        # 3. Setup time discretization
        timesteps = sampling_schedule_grid(sampling_schedule, steps, eps, device)
        dt = (1 - eps) / steps

        # 4. Execute main SMC loop
        x, log_weights, trace_containers = _execute_smc_loop(
            predictor,
            cfg_score_fn,
            x,
            log_weights,
            cond,
            timesteps,
            dt,
            proposal_strength,
            weight_computer,
            resampling_params,
            trace_containers,
            projector,
        )

        # 5. Optional denoising step
        if denoise:
            x, log_weights, trace_containers = _handle_denoising(
                x,
                log_weights,
                cfg_score_fn,
                cond,
                timesteps,
                denoiser,
                projector,
                resampling_params,
                trace_containers,
            )

        # 6. Package and return results
        return SMCOutput(
            particles=x,
            ess_trace=trace_containers.get("ess_trace"),
            weight_trace=trace_containers.get("weight_trace"),
            resampling_trace=trace_containers.get("resampling_trace"),
            particles_trace=trace_containers.get("particles_trace"),
        )

    return smc_sampler


def get_cfg_rne_sampler(
    graph: graph.Graph,
    noise: noise.Noise,
    batch_dims: Tuple[int, ...],
    predictor: Predictor,
    steps: int,
    proposal_strength: float = 1.0,
    smc_temperature: float = 1.0,
    denoise: bool = True,
    eps: float = 1e-5,
    ess_threshold: float = 0.5,
    resample_fraction: float = 0.2,
    sampling_threshold: float = 0.1,
    resampling_method: str = "partial",
    device: torch.device = torch.device("cpu"),
    proj_fun: Callable[
        [Int[torch.Tensor, "batch datadim"]], Int[torch.Tensor, "batch datadim"]
    ] = lambda x: x,
    sampling_schedule: str = "linear",
    method: str = "score",
) -> Callable:
    """Create an RNE sampler for unbiased CFG sampling using Radon-Nikodym Estimator.

    This sampler implements Sequential Monte Carlo with RNE weight computation to sample
    from the true tempered distribution, avoiding the bias introduced by standard CFG.

    Args:
        graph: Rate matrix graph (Uniform or Absorbing)
        noise: Noise schedule (LogLinear, Geometric, etc.)
        batch_dims: Dimensions for initial particle sampling
        predictor: Predictor algorithm (should be EulerPredictor for RNE support)
        steps: Number of diffusion steps
        proposal_strength: CFG temperature for proposal distribution q
        smc_temperature: Temperature α for likelihood ratio weighting
        denoise: Whether to perform final denoising step
        eps: Final time (close to 0)
        ess_threshold: ESS threshold for triggering resampling (fraction of particle count)
        resample_fraction: Fraction of particles to resample when triggered
        sampling_threshold: Don't resample before this fraction of steps
        resampling_method: Resampling strategy ("partial", "multinomial", "stratified")
        device: Device for computation
        proj_fun: Projection function (identity by default)
        sampling_schedule: Time discretization schedule ("linear" or "cosine")

    Returns:
        RNE sampler function that takes (uncond_model, cond_model, cond)
    """
    from lib.resampling import get_resampling_fn
    from lib.smc_weights import RNEWeightComputer

    # Validation
    if not hasattr(predictor, "_sample_from_cfg_proposal"):
        raise ValueError(
            "RNE sampler requires predictor with _sample_from_cfg_proposal method"
        )

    # Setup shared components
    projector = proj_fun
    denoiser = Denoiser(graph, noise)
    weight_computer = RNEWeightComputer(graph, noise, smc_temperature=smc_temperature)
    resampling_fn = get_resampling_fn(resampling_method)

    # Package resampling parameters
    resampling_params = {
        "method": resampling_method,
        "ess_threshold": ess_threshold,
        "fraction": resample_fraction,
        "sampling_threshold": sampling_threshold,
        "resampling_fn": resampling_fn,
    }

    @torch.no_grad()
    def rne_sampler(
        uncond_model: torch.nn.Module,
        cond_model: torch.nn.Module,
        cond: torch.Tensor,
        traces: List[str] = None,
    ) -> SMCOutput:
        """RNE sampler for unbiased CFG.

        Args:
            uncond_model: Unconditional diffusion model
            cond_model: Conditional diffusion model
            cond: Conditioning variable (e.g., class labels [batch] or token sequences [batch, datadim])
            traces: List of traces to collect. Options: ["ess", "weight", "resampling", "particles"]

        Returns:
            SMCOutput dataclass containing final particles and requested traces
        """
        # 1. Initialize SMC state
        x, log_weights, trace_containers = _initialize_smc_state(
            graph, batch_dims, device, traces or [], steps
        )

        # 2. Create CFG score function
        cfg_score_fn = mutils.get_cfg_score_fn(
            uncond_model, cond_model, proposal_strength, train=False, sampling=True
        )

        # 3. Setup time discretization
        timesteps = sampling_schedule_grid(sampling_schedule, steps, eps, device)
        dt = (1 - eps) / steps

        # 4. Execute main RNE loop
        for i in range(steps):
            # Prepare for this step
            t = timesteps[i] * torch.ones(x.shape[0], device=x.device)
            x = projector(x)

            # Perform RNE step
            x_next, log_weight_update = _rne_step(
                predictor,
                cfg_score_fn,
                x,
                cond,
                t,
                dt,
                proposal_strength,
                weight_computer,
                device,
                method,
            )
            log_weights += log_weight_update

            # Update traces
            trace_containers = _update_traces(trace_containers, x_next, log_weights, i)

            # Handle adaptive resampling
            x_next, log_weights, ess, trace_containers = _handle_resampling(
                x_next, log_weights, i, steps, resampling_params, trace_containers
            )

            x = x_next

        # 5. Optional denoising step
        if denoise:
            x, log_weights, trace_containers = _handle_denoising(
                x,
                log_weights,
                cfg_score_fn,
                cond,
                timesteps,
                denoiser,
                projector,
                resampling_params,
                trace_containers,
            )

        # 6. Package and return results
        return SMCOutput(
            particles=x,
            ess_trace=trace_containers.get("ess_trace"),
            weight_trace=trace_containers.get("weight_trace"),
            resampling_trace=trace_containers.get("resampling_trace"),
            particles_trace=trace_containers.get("particles_trace"),
        )

    return rne_sampler


def get_pt_sampler(
    graph: graph.Graph,
    noise: noise.Noise,
    time_steps: Float[torch.Tensor, "time_steps"],
    step_size: float,
    smc_temperature: float,
    num_steps: int,
    burn_in_steps: int,
    cfg_temperature: float = 1.0,
    store_k_paths: int = 0,
    store_every_n_steps: int = 1,
    num_local_steps: int = 0,
    force_swap_at_one: bool = False,
    method: str = "score",
    score_fn_type: str = "standard",
    keep_burn_in: bool = False,
    batch_size_pt: Optional[int] = None,
) -> Callable:
    """Create a PT sampler - uses parallel tempering with Metropolis acceptance.

    This sampler implements Parallel Tempering to sample from paths by evolving
    pairs along the time trajectory and using Metropolis acceptance for swapping.

    Args:
        graph: Rate matrix graph (Uniform or Absorbing)
        noise: Noise schedule (LogLinear, Geometric, etc.)
        time_steps: Time discretization points for the path [time_steps]
        step_size: Time step size between adjacent points
        smc_temperature: Temperature α for RNE weight computation
        num_steps: Number of PT iterations to perform
        burn_in_steps: Number of initial steps to discard
        cfg_temperature: Classifier-free guidance strength
        store_k_paths: Number of paths to store at intervals (0 = disabled)
        store_every_n_steps: Store a path every n steps
        num_local_steps: Number of local CTMC moves after accepted Metropolis updates (default: 0)
        force_swap_at_one: Force all swaps to be accepted for debugging (default: False)
        method: log_R computation method ("score" or "prob")
        score_fn_type: Score function type ("standard", "discdiff", or "finetune")
        keep_burn_in: Whether to keep samples during burn-in period (default: False)
        batch_size_pt: Maximum batch size for processing PT pairs to avoid OOM (None = process all pairs at once)

    Returns:
        PT sampler function that takes (uncond_model, cond_model, cond, initial_path)
    """

    @torch.no_grad()
    def pt_sampler(
        uncond_model: torch.nn.Module,
        cond_model: torch.nn.Module,
        cond: Float[torch.Tensor, "1 ..."],
        initial_path: Int[torch.Tensor, "time_steps datadim"],
        traces: List[str] = None,
    ) -> PTOutput:
        """PT sampler for path evolution.

        Args:
            uncond_model: Unconditional diffusion model
            cond_model: Conditional diffusion model
            cond: Conditioning variable (e.g., class labels [1, 1] or token sequences [1, datadim])
            initial_path: Initial path to evolve [time_steps, datadim]
            traces: List of traces to collect. Options: ["acceptance_rates", "path_trace"]

        Returns:
            PTOutput dataclass containing samples, final path, and requested traces
        """
        # Import here to avoid circular dependency
        from lib.pt import get_parallel_tempering

        # Create CFG score function based on type
        if score_fn_type == "discdiff":
            cfg_score_fn = mutils.get_score_discdiff_fn(
                uncond_model, cond_model, cfg_temperature, train=False, sampling=True
            )
        elif score_fn_type == "finetune":
            cfg_score_fn = mutils.get_cfg_score_finetune_fn(
                uncond_model, cond_model, cfg_temperature, train=False, sampling=True
            )
        else:  # standard
            cfg_score_fn = mutils.get_cfg_score_fn(
                uncond_model, cond_model, cfg_temperature, train=False, sampling=True
            )

        assert cond.shape[0] == 1, (
            "cond must have batch dimension 1. Only a single condition can be passed in PT sampling."
        )

        # Create PT function with fixed parameters
        pt_fn = get_parallel_tempering(
            time_steps,
            step_size,
            graph,
            noise,
            cfg_score_fn,
            smc_temperature,
            num_local_steps,
            force_swap_at_one,
            method,
            keep_burn_in,
            batch_size_pt,
        )

        # Run parallel tempering sampling
        pt_output = pt_fn(
            initial_path,
            num_steps,
            burn_in_steps,
            cond,
            traces,
            store_k_paths,
            store_every_n_steps,
        )

        return pt_output

    return pt_sampler


def local_move(
    graph: graph.Graph,
    xt: Int[torch.Tensor, "batch datadim"],
    score: Float[torch.Tensor, "batch datadim vocab"],
    sigma: Float[torch.Tensor, "batch"],
) -> Int[torch.Tensor, "batch datadim"]:
    """
    Local move according to the rate R_t + Q_t where R is the forward rate and Q its reverse rate.
    This corresponds to sampling from the stationary distribution of the CTMC. c.f. 4.4 in https://proceedings.neurips.cc/paper_files/paper/2022/file/b5b528767aa35f5b1a60fe0aaeca0563-Paper-Conference.pdf
    """

    rate: Float[torch.Tensor, "batch datadim vocab"] = graph.rate(xt)
    reverse_rate: Float[torch.Tensor, "batch datadim vocab"] = graph.reverse_rate(
        xt, score
    )

    stationary_rate: Float[torch.Tensor, "batch datadim vocab"] = rate + reverse_rate
    stationary_rate = stationary_rate * sigma[..., None, None]

    xt_next: Int[torch.Tensor, "batch datadim"]
    xt_next, _ = graph.sample_rate(xt, stationary_rate)
    return xt_next
