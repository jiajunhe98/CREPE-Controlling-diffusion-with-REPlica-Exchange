import time
from typing import Callable, List, Optional

import torch
from jaxtyping import Float, Int

from lib.graph import Graph
from lib.noise import Noise
from lib.sampler import EulerPredictor
from lib.smc_weights import RNEWeightComputer


def create_pairs_from_path(
    Y_Path: Int[torch.Tensor, "time_steps datadim"],
    time_steps: Float[torch.Tensor, "time_steps"],
    pt_iteration: int,
) -> tuple[Int[torch.Tensor, "npairs 2 datadim"], Float[torch.Tensor, "npairs"], int]:
    """
    Extract adjacent pairs from path for current PT step.

    Args:
        Y_Path: Current path state [time_steps, datadim]
        time_steps: Time discretization points [time_steps]
        pt_iteration: Current iteration index in PT loop

    Returns:
        pairs: Adjacent pairs for processing [npairs, 2, datadim]
        t_pairs: Corresponding time steps [npairs]
        min_len: Number of pairs created
    """
    min_len = min(
        len(Y_Path[pt_iteration % 2 :: 2]), len(Y_Path[pt_iteration % 2 + 1 :: 2])
    )
    pairs = torch.stack(
        [
            Y_Path[pt_iteration % 2 :: 2][:min_len],
            Y_Path[pt_iteration % 2 + 1 :: 2][:min_len],
        ],
        dim=1,
    )
    t_pairs = time_steps[pt_iteration % 2 + 1 :: 2][:min_len]

    assert len(t_pairs) == pairs.shape[0], (
        f"t_pairs: {t_pairs.shape}, pairs: {pairs.shape}"
    )

    return pairs, t_pairs, min_len


def compute_metropolis_acceptance(
    log_weights_reverse: Float[torch.Tensor, "npairs"],
    log_weights_forward: Float[torch.Tensor, "npairs"],
) -> tuple[Float[torch.Tensor, "npairs"], torch.Tensor]:
    """
    Compute Metropolis acceptance probabilities from log weights.

    Args:
        log_weights_reverse: Reverse-time log weights [npairs]
        log_weights_forward: Forward-time log weights [npairs]

    Returns:
        acceptance_probability: Clamped acceptance probabilities [npairs]
        accept_mask: Boolean mask for accepted proposals [npairs]
    """
    log_weights_sum = log_weights_reverse + log_weights_forward
    exp_weights = torch.exp(log_weights_sum)
    acceptance_probability = torch.clamp(exp_weights, max=1.0, min=0.0)

    accept_mask = (
        torch.rand(
            acceptance_probability.shape[0], device=acceptance_probability.device
        )
        < acceptance_probability
    )

    return acceptance_probability, accept_mask


def apply_metropolis_update(
    pairs: Int[torch.Tensor, "npairs 2 datadim"],
    X_later: Int[torch.Tensor, "npairs datadim"],
    X_earlier: Int[torch.Tensor, "npairs datadim"],
    accept_mask: torch.Tensor,
) -> Int[torch.Tensor, "npairs 2 datadim"]:
    """
    Apply Metropolis acceptance to update pairs.

    Args:
        pairs: Current pairs [npairs, 2, datadim]
        X_later: Proposed later states [npairs, datadim]
        X_earlier: Proposed earlier states [npairs, datadim]
        accept_mask: Acceptance mask [npairs]

    Returns:
        Updated pairs with accepted proposals [npairs, 2, datadim]
    """
    new_pairs = torch.stack([X_later, X_earlier], dim=1)
    return torch.where(accept_mask[:, None, None], new_pairs, pairs)


def apply_local_moves(
    pairs: Int[torch.Tensor, "npairs 2 datadim"],
    accept_mask: torch.Tensor,  # [npairs] bool
    t_pairs: Float[torch.Tensor, "npairs"],
    cond: torch.Tensor,
    num_local_steps: int,
    graph: Graph,
    noise: Noise,
    cfg_score_fn: Callable,
) -> Int[torch.Tensor, "npairs 2 datadim"]:
    """
    Apply local CTMC moves to accepted pairs after Metropolis update.

    Args:
        pairs: Current pairs [npairs, 2, datadim]
        accept_mask: Boolean mask for accepted proposals [npairs]
        t_pairs: Time steps for pairs [npairs]
        cond: Conditioning variable
        num_local_steps: Number of local move iterations
        graph: Rate matrix graph
        noise: Noise schedule
        cfg_score_fn: CFG score function

    Returns:
        Updated pairs with local moves applied to accepted pairs [npairs, 2, datadim]
    """
    if num_local_steps == 0:
        return pairs

    from lib.sampler import local_move

    # Get accepted pairs for local moves
    accepted_pairs: Int[torch.Tensor, "num_accepted 2 datadim"] = pairs[accept_mask]

    if len(accepted_pairs) == 0:
        return pairs

    # Get time and noise parameters for accepted pairs
    t_accepted: Float[torch.Tensor, "num_accepted"] = t_pairs[accept_mask]
    sigma_accepted: Float[torch.Tensor, "num_accepted"]
    sigma_accepted, _ = noise(t_accepted)

    # Apply local moves to both X_later and X_earlier positions
    for local_step in range(num_local_steps):
        # Local move for X_later (position 0)
        X_later_accepted: Int[torch.Tensor, "num_accepted datadim"] = accepted_pairs[
            :, 0, :
        ]

        # Get scores for local moves
        proposal_score_local: Float[torch.Tensor, "num_accepted datadim vocab"]
        cond_expanded = cond.expand(X_later_accepted.shape[0], *cond.shape[1:])
        proposal_score_local, (_, _) = cfg_score_fn(
            X_later_accepted, cond_expanded, sigma_accepted, with_aux=True
        )

        # Apply local move to X_later
        X_later_local: Int[torch.Tensor, "num_accepted datadim"] = local_move(
            graph, X_later_accepted, proposal_score_local, sigma_accepted
        )
        accepted_pairs[:, 0, :] = X_later_local

        # Local move for X_earlier (position 1)
        X_earlier_accepted: Int[torch.Tensor, "num_accepted datadim"] = accepted_pairs[
            :, 1, :
        ]
        cond_expanded = cond.expand(X_earlier_accepted.shape[0], *cond.shape[1:])
        # Get scores for local moves
        proposal_score_local, (_, _) = cfg_score_fn(
            X_earlier_accepted, cond_expanded, sigma_accepted, with_aux=True
        )

        # Apply local move to X_earlier
        X_earlier_local: Int[torch.Tensor, "num_accepted datadim"] = local_move(
            graph, X_earlier_accepted, proposal_score_local, sigma_accepted
        )
        accepted_pairs[:, 1, :] = X_earlier_local

    # Update the pairs with locally moved accepted pairs
    pairs[accept_mask] = accepted_pairs

    return pairs


def reconstruct_path_from_pairs(
    Y_Path: Int[torch.Tensor, "time_steps datadim"],
    pairs: Int[torch.Tensor, "npairs 2 datadim"],
    pt_iteration: int,
    min_len: int,
) -> None:
    """
    Update path with evolved pairs (modifies Y_Path in-place).

    Args:
        Y_Path: Path to update (modified in-place) [time_steps, datadim]
        pairs: Updated pairs [npairs, 2, datadim]
        pt_iteration: Current iteration index
        min_len: Number of pairs to update
    """
    Y_Path[pt_iteration % 2 :: 2][:min_len] = pairs[:, 0, :]
    Y_Path[pt_iteration % 2 + 1 :: 2][:min_len] = pairs[:, 1, :]


class PTTraceCollector:
    """
    Manages trace collection for parallel tempering sampling.

    Handles acceptance statistics, log weights per timestep, path traces,
    and k-path storage during PT sampling.
    """

    def __init__(
        self,
        traces: List[str],
        time_steps: Float[torch.Tensor, "time_steps"],
        Y_Path: Int[torch.Tensor, "time_steps datadim"],
        store_k_paths: int = 0,
        store_every_n_steps: int = 1,
    ):
        self.traces = traces or []
        self.time_steps = time_steps
        self.device = Y_Path.device

        # Initialize acceptance counting
        self.accept_count_per_timestep = None
        if "accept_mask" in self.traces:
            self.accept_count_per_timestep = torch.zeros(
                len(time_steps), dtype=torch.long, device=self.device
            )

        # Initialize log weight accumulation
        self.log_weights_reverse_per_timestep = None
        self.log_weights_forward_per_timestep = None
        self.weights_count_per_timestep = None
        if "log_weights_reverse" in self.traces or "log_weights_forward" in self.traces:
            self.log_weights_reverse_per_timestep = torch.zeros(
                len(time_steps), dtype=torch.float, device=self.device
            )
            self.log_weights_forward_per_timestep = torch.zeros(
                len(time_steps), dtype=torch.float, device=self.device
            )
            self.weights_count_per_timestep = torch.zeros(
                len(time_steps), dtype=torch.long, device=self.device
            )

        # Initialize path tracing
        self.path_trace = [] if "path" in self.traces else None

        # Initialize k-path storage
        self.k_paths_storage = None
        self.k_paths_counter = 0
        self.store_every_n_steps = store_every_n_steps
        if store_k_paths > 0:
            # Store k-paths on CPU to save GPU memory during debugging
            self.k_paths_storage = torch.zeros(
                store_k_paths,
                *Y_Path.shape,
                dtype=Y_Path.dtype,
                device=torch.device("cpu"),
            )

    def update_acceptance_traces(
        self, accept_mask: torch.Tensor, pt_iteration: int, min_len: int
    ) -> None:
        """Update acceptance traces with current step results."""
        if self.accept_count_per_timestep is not None:
            time_indices = torch.arange(
                pt_iteration % 2, len(self.time_steps), 2, device=self.device
            )[:min_len]
            self.accept_count_per_timestep[time_indices] += accept_mask.long()

    def update_weight_traces(
        self,
        log_weights_reverse: Float[torch.Tensor, "npairs"],
        log_weights_forward: Float[torch.Tensor, "npairs"],
        pt_iteration: int,
        min_len: int,
    ) -> None:
        """Update log weight traces with current step results."""
        if self.log_weights_reverse_per_timestep is not None:
            time_indices = torch.arange(
                pt_iteration % 2, len(self.time_steps), 2, device=self.device
            )[:min_len]
            self.log_weights_reverse_per_timestep[time_indices] += log_weights_reverse
            self.log_weights_forward_per_timestep[time_indices] += log_weights_forward
            self.weights_count_per_timestep[time_indices] += 1

    def update_path_trace(
        self, Y_Path: Int[torch.Tensor, "time_steps datadim"]
    ) -> None:
        """Store current path state if path tracing is enabled."""
        if self.path_trace is not None:
            self.path_trace.append(Y_Path.clone().cpu())

    def maybe_store_k_path(
        self, Y_Path: Int[torch.Tensor, "time_steps datadim"], pt_iteration: int
    ) -> None:
        """Store path in k-path circular buffer if conditions are met."""
        if (
            self.k_paths_storage is not None
            and pt_iteration % self.store_every_n_steps == 0
        ):
            buffer_idx = self.k_paths_counter % self.k_paths_storage.shape[0]
            # Store on CPU to save GPU memory during debugging
            self.k_paths_storage[buffer_idx] = Y_Path.clone().cpu()
            self.k_paths_counter += 1


def get_parallel_tempering(
    time_steps: Float[torch.Tensor, "time_steps"],
    step_size: float,
    graph: Graph,
    noise: Noise,
    cfg_score_fn: Callable,
    smc_temperature: float,
    num_local_steps: int = 0,
    force_swap_at_one: bool = False,
    method: str = "score",
    keep_burn_in: bool = False,
    batch_size_pt: Optional[int] = None,
):
    """
    Creates a parallel tempering sampler with fixed parameters.

    Args:
        time_steps: Time discretization points [time_steps]
        step_size: Time step size between adjacent points
        graph: Rate matrix graph
        noise: Noise schedule
        cfg_score_fn: CFG score function
        smc_temperature: Temperature for RNE weight computation
        num_local_steps: Number of local CTMC moves after accepted Metropolis updates (default: 0)
        force_swap_at_one: Force all swaps to be accepted for debugging (default: False)
        method: log_R computation method ("score" or "prob")
        keep_burn_in: Whether to keep samples during burn-in period (default: False)
        batch_size_pt: Maximum batch size for processing PT pairs to avoid OOM (None = process all pairs at once)

    Returns a sample_parallel_tempering function that can be called
    with different Y_Path, num_steps, burn_in_steps, and cond.
    """
    predictor = EulerPredictor(graph, noise)
    weight_computer = RNEWeightComputer(graph, noise, smc_temperature)

    def _sample_parallel_tempering_swap_at_one(
        Y_Path: Int[torch.Tensor, "time_steps datadim"],
        num_steps: int,
        burn_in_steps: int,
        cond: torch.Tensor,
        traces: Optional[List[str]] = None,
        store_k_paths: int = 0,
        store_every_n_steps: int = 1,
    ):
        """
        Debug version of parallel tempering that forces all swaps to be accepted.

        This bypasses weight computation entirely and always accepts all proposed swaps.
        Used for debugging purposes to isolate swap acceptance from weight computation.
        """
        samples = []
        burn_in_samples = [] if keep_burn_in else None
        path_length = len(Y_Path)
        store_on_odd = path_length % 2 == 1

        # Initialize trace collection (but skip weight-related traces since we don't compute them)
        debug_traces = [
            t
            for t in (traces or [])
            if t not in ["log_weights_reverse", "log_weights_forward"]
        ]
        trace_collector = PTTraceCollector(
            debug_traces, time_steps, Y_Path, store_k_paths, store_every_n_steps
        )

        for i in range(num_steps):
            # Extract pairs from path for current PT step
            pairs, t_pairs, min_len = create_pairs_from_path(Y_Path, time_steps, i)

            # Process pairs in batches if batch_size_pt is specified
            if batch_size_pt is None or min_len <= batch_size_pt:
                # No batching needed - process all pairs at once
                Y_later = pairs[:, 0, :]

                X_earlier, proposal_score_1, uncond_score_1, cond_score_1 = (
                    predictor._sample_from_cfg_proposal(
                        cfg_score_fn=cfg_score_fn,
                        x=Y_later,
                        cond=cond,
                        t=t_pairs,  # t - (k+1)dt here.
                        step_size=step_size,
                        dtype=Y_later.dtype,
                    )
                )

                # Evolve Y_{1 - (k+1)dt} -> X_{1 - kdt} using the noising process
                Y_earlier: Int[torch.Tensor, "npairs datadim"] = pairs[:, 1, :]

                sigma, d_sigma = noise(t_pairs)
                rate = graph.rate(Y_earlier)
                X_later, _ = graph.sample_rate(
                    Y_earlier, step_size * d_sigma[..., None, None] * rate
                )
            else:
                # Batch processing to avoid OOM
                num_batches = (min_len + batch_size_pt - 1) // batch_size_pt
                X_earlier_batches = []
                X_later_batches = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size_pt
                    end_idx = min(start_idx + batch_size_pt, min_len)

                    # Extract batch
                    pairs_batch = pairs[start_idx:end_idx]
                    t_pairs_batch = t_pairs[start_idx:end_idx]

                    # Evolve Y_{1 - kdt} -> X_{1 - (k+1)dt} using the proposal
                    Y_later_batch = pairs_batch[:, 0, :]

                    X_earlier_batch, _, _, _ = predictor._sample_from_cfg_proposal(
                        cfg_score_fn=cfg_score_fn,
                        x=Y_later_batch,
                        cond=cond,
                        t=t_pairs_batch,
                        step_size=step_size,
                        dtype=Y_later_batch.dtype,
                    )
                    X_earlier_batches.append(X_earlier_batch)

                    # Evolve Y_{1 - (k+1)dt} -> X_{1 - kdt} using the noising process
                    Y_earlier_batch = pairs_batch[:, 1, :]

                    sigma_batch, d_sigma_batch = noise(t_pairs_batch)
                    rate_batch = graph.rate(Y_earlier_batch)
                    X_later_batch, _ = graph.sample_rate(
                        Y_earlier_batch,
                        step_size * d_sigma_batch[..., None, None] * rate_batch,
                    )
                    X_later_batches.append(X_later_batch)

                # Concatenate batches
                X_earlier = torch.cat(X_earlier_batches, dim=0)
                X_later = torch.cat(X_later_batches, dim=0)

            # Force all swaps to be accepted (no weight computation needed)
            accept_mask = torch.ones(min_len, dtype=torch.bool, device=Y_Path.device)

            # Apply Metropolis update to pairs (will accept all)
            pairs = apply_metropolis_update(pairs, X_later, X_earlier, accept_mask)

            # Apply local moves to accepted pairs if num_local_steps > 0
            pairs = apply_local_moves(
                pairs,
                accept_mask,
                t_pairs,
                cond,
                num_local_steps,
                graph,
                noise,
                cfg_score_fn,
            )

            # Update traces (skip weight traces since we don't compute them)
            trace_collector.update_acceptance_traces(accept_mask, i, min_len)
            # Skip weight traces: trace_collector.update_weight_traces(...)
            trace_collector.update_path_trace(Y_Path)
            trace_collector.maybe_store_k_path(Y_Path, i)

            # Update path with evolved pairs
            reconstruct_path_from_pairs(Y_Path, pairs, i, min_len)

            # Store samples during burn-in if requested
            if (
                keep_burn_in
                and i < burn_in_steps
                and ((store_on_odd and i % 2 == 1) or (not store_on_odd and i % 2 == 0))
            ):
                burn_in_samples.append(Y_Path[-1].clone())

            # Store sample after burn-in
            elif i >= burn_in_steps and (
                (store_on_odd and i % 2 == 1) or (not store_on_odd and i % 2 == 0)
            ):
                samples.append(Y_Path[-1].clone())

        # Import PTOutput here to avoid circular import
        from lib.sampler import PTOutput

        return PTOutput(
            samples=samples,
            final_path=Y_Path,
            burn_in_samples=burn_in_samples,
            acceptance_rates=None,  # Could be computed from accept_count_per_timestep if needed
            path_trace=trace_collector.path_trace,
            accept_count_per_timestep=trace_collector.accept_count_per_timestep,
            log_weights_reverse_per_timestep=None,  # Not computed in force swap mode
            log_weights_forward_per_timestep=None,  # Not computed in force swap mode
            weights_count_per_timestep=None,  # Not computed in force swap mode
            k_paths_storage=trace_collector.k_paths_storage,
        )

    def sample_parallel_tempering(
        Y_Path: Int[torch.Tensor, "time_steps datadim"],
        num_steps: int,
        burn_in_steps: int,
        cond: Int[torch.Tensor, "1 ..."],
        traces: Optional[List[str]] = None,
        store_k_paths: int = 0,
        store_every_n_steps: int = 1,
    ):
        """
        Performs parallel tempering sampling.

        Y_path is a tensors of shape (time_steps, datadim). that represents [Y_1, Y_1 - dt, Y_1 - 2dt, ..., Y_0].
        That path comes from some proposal, such as CFG for example.

        Args:
            store_k_paths: Number of paths to store at intervals (0 = disabled)
            store_every_n_steps: Store a path every n steps
        """
        # Use debug version if force_swap_at_one is enabled
        if force_swap_at_one:
            return _sample_parallel_tempering_swap_at_one(
                Y_Path,
                num_steps,
                burn_in_steps,
                cond,
                traces,
                store_k_paths,
                store_every_n_steps,
            )
        samples = []
        burn_in_samples = [] if keep_burn_in else None
        path_length = len(Y_Path)
        store_on_odd = path_length % 2 == 1

        # Initialize trace collection
        trace_collector = PTTraceCollector(
            traces, time_steps, Y_Path, store_k_paths, store_every_n_steps
        )

        for i in range(num_steps):
            # DEBUG: Step separator
            # from printgrave import print_step_separator
            # print_step_separator(i, "PT Iteration")

            # Extract pairs from path for current PT step
            pairs, t_pairs, min_len = create_pairs_from_path(Y_Path, time_steps, i)

            # Process pairs in batches if batch_size_pt is specified
            if batch_size_pt is None or min_len <= batch_size_pt:
                # No batching needed - process all pairs at once
                Y_later = pairs[:, 0, :]

                cond_expanded = cond.expand(Y_later.shape[0], *cond.shape[1:])
                X_earlier, proposal_score_1, uncond_score_1, cond_score_1 = (
                    predictor._sample_from_cfg_proposal(
                        cfg_score_fn=cfg_score_fn,
                        x=Y_later,
                        cond=cond_expanded,
                        t=t_pairs,  # t - (k+1)dt here.
                        step_size=step_size,
                        dtype=Y_later.dtype,
                    )
                )

                # # DEBUG: Check scores before weight computation
                # from printgrave import print_tensor_stats
                # print_tensor_stats(
                #     uncond_score_1, f"uncond_score_1 (step {i})", show_values=False
                # )
                # print_tensor_stats(
                #     cond_score_1, f"cond_score_1 (step {i})", show_values=False
                # )
                # print_tensor_stats(
                #     proposal_score_1, f"proposal_score_1 (step {i})", show_values=False
                # )

                # Compute the weights on (X_{1 - (k+1)dt}, Y_{1 - kdt})
                log_weights_reverse_time: Float[torch.Tensor, "npairs"] = (
                    weight_computer.compute_rne_weight_update(
                        x_earlier=X_earlier,
                        x_later=Y_later,
                        t=t_pairs,  # t - (k+1)dt here.
                        dt=step_size,
                        uncond_score=uncond_score_1,
                        cond_score=cond_score_1,
                        proposal_score=proposal_score_1,
                        device=Y_later.device,
                        method=method,
                    )
                )

                # Evolve Y_{1 - (k+1)dt} -> X_{1 - kdt} using the noising process
                Y_earlier: Int[torch.Tensor, "npairs datadim"] = pairs[:, 1, :]

                sigma, d_sigma = noise(t_pairs)

                rate = graph.rate(Y_earlier)
                X_later, _ = graph.sample_rate(
                    Y_earlier, step_size * d_sigma[..., None, None] * rate
                )
                cond_expanded = cond.expand(X_later.shape[0], *cond.shape[1:])

                proposal_score_2, (uncond_score_2, cond_score_2) = cfg_score_fn(
                    X_later, cond_expanded, sigma, with_aux=True
                )

                # DEBUG: Check second set of scores
                # print_tensor_stats(
                #     uncond_score_2, f"uncond_score_2 (step {i})", show_values=False
                # )
                # print_tensor_stats(
                #     cond_score_2, f"cond_score_2 (step {i})", show_values=False
                # )
                # print_tensor_stats(
                #     proposal_score_2, f"proposal_score_2 (step {i})", show_values=False
                # )

                # DEBUG: Check element-wise equality between X_later and Y_later
                # from printgrave import print_tensor_equality
                # print_tensor_equality(X_later, Y_later, "X_later", "Y_later", step=i)

                # DEBUG: Check element-wise equality between X_earlier and Y_earlier
                # print_tensor_equality(
                #     X_earlier, Y_earlier, "X_earlier", "Y_earlier", step=i
                # )

                log_weights_forward_time: Float[
                    torch.Tensor, "npairs"
                ] = -weight_computer.compute_rne_weight_update(
                    x_earlier=Y_earlier,
                    x_later=X_later,
                    t=t_pairs,
                    dt=step_size,
                    uncond_score=uncond_score_2,
                    cond_score=cond_score_2,
                    proposal_score=proposal_score_2,
                    device=Y_earlier.device,
                    method=method,
                )
            else:
                # Batch processing to avoid OOM
                num_batches = (min_len + batch_size_pt - 1) // batch_size_pt
                X_earlier_batches = []
                X_later_batches = []
                log_weights_reverse_batches = []
                log_weights_forward_batches = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size_pt
                    end_idx = min(start_idx + batch_size_pt, min_len)

                    # Extract batch
                    pairs_batch = pairs[start_idx:end_idx]
                    t_pairs_batch = t_pairs[start_idx:end_idx]

                    # First evolution: Y_{1 - kdt} -> X_{1 - (k+1)dt}
                    Y_later_batch = pairs_batch[:, 0, :]
                    cond_expanded_batch = cond.expand(
                        Y_later_batch.shape[0], *cond.shape[1:]
                    )

                    (
                        X_earlier_batch,
                        proposal_score_1_batch,
                        uncond_score_1_batch,
                        cond_score_1_batch,
                    ) = predictor._sample_from_cfg_proposal(
                        cfg_score_fn=cfg_score_fn,
                        x=Y_later_batch,
                        cond=cond_expanded_batch,
                        t=t_pairs_batch,
                        step_size=step_size,
                        dtype=Y_later_batch.dtype,
                    )

                    # Compute reverse time weights
                    log_weights_reverse_batch = (
                        weight_computer.compute_rne_weight_update(
                            x_earlier=X_earlier_batch,
                            x_later=Y_later_batch,
                            t=t_pairs_batch,
                            dt=step_size,
                            uncond_score=uncond_score_1_batch,
                            cond_score=cond_score_1_batch,
                            proposal_score=proposal_score_1_batch,
                            device=Y_later_batch.device,
                            method=method,
                        )
                    )

                    # Second evolution: Y_{1 - (k+1)dt} -> X_{1 - kdt}
                    Y_earlier_batch = pairs_batch[:, 1, :]
                    sigma_batch, d_sigma_batch = noise(t_pairs_batch)
                    rate_batch = graph.rate(Y_earlier_batch)
                    X_later_batch, _ = graph.sample_rate(
                        Y_earlier_batch,
                        step_size * d_sigma_batch[..., None, None] * rate_batch,
                    )

                    cond_expanded_batch = cond.expand(
                        X_later_batch.shape[0], *cond.shape[1:]
                    )
                    (
                        proposal_score_2_batch,
                        (
                            uncond_score_2_batch,
                            cond_score_2_batch,
                        ),
                    ) = cfg_score_fn(
                        X_later_batch, cond_expanded_batch, sigma_batch, with_aux=True
                    )

                    # Compute forward time weights
                    log_weights_forward_batch = (
                        -weight_computer.compute_rne_weight_update(
                            x_earlier=Y_earlier_batch,
                            x_later=X_later_batch,
                            t=t_pairs_batch,
                            dt=step_size,
                            uncond_score=uncond_score_2_batch,
                            cond_score=cond_score_2_batch,
                            proposal_score=proposal_score_2_batch,
                            device=Y_earlier_batch.device,
                            method=method,
                        )
                    )

                    # Store batch results
                    X_earlier_batches.append(X_earlier_batch)
                    X_later_batches.append(X_later_batch)
                    log_weights_reverse_batches.append(log_weights_reverse_batch)
                    log_weights_forward_batches.append(log_weights_forward_batch)

                # Concatenate all batches
                X_earlier = torch.cat(X_earlier_batches, dim=0)
                X_later = torch.cat(X_later_batches, dim=0)
                log_weights_reverse_time = torch.cat(log_weights_reverse_batches, dim=0)
                log_weights_forward_time = torch.cat(log_weights_forward_batches, dim=0)

            # DEBUG: Print weight values to investigate infinities
            # from printgrave import print_pt_weights_debug
            # print_pt_weights_debug(
            #     log_weights_reverse_time, log_weights_forward_time, i, min_len
            # )

            # Compute Metropolis acceptance probabilities
            acceptance_probability, accept_mask = compute_metropolis_acceptance(
                log_weights_reverse_time, log_weights_forward_time
            )

            # Apply Metropolis update to pairs
            pairs = apply_metropolis_update(pairs, X_later, X_earlier, accept_mask)

            # Apply local moves to accepted pairs if num_local_steps > 0
            pairs = apply_local_moves(
                pairs,
                accept_mask,
                t_pairs,
                cond,
                num_local_steps,
                graph,
                noise,
                cfg_score_fn,
            )

            # Update all traces
            trace_collector.update_acceptance_traces(accept_mask, i, min_len)
            trace_collector.update_weight_traces(
                log_weights_reverse_time, log_weights_forward_time, i, min_len
            )
            trace_collector.update_path_trace(Y_Path)
            trace_collector.maybe_store_k_path(Y_Path, i)

            # Update path with evolved pairs
            reconstruct_path_from_pairs(Y_Path, pairs, i, min_len)

            # Store samples during burn-in if requested
            if (
                keep_burn_in
                and i < burn_in_steps
                and ((store_on_odd and i % 2 == 1) or (not store_on_odd and i % 2 == 0))
            ):
                burn_in_samples.append(Y_Path[-1].clone())

            # Store sample after burn-in
            elif i >= burn_in_steps and (
                (store_on_odd and i % 2 == 1) or (not store_on_odd and i % 2 == 0)
            ):
                samples.append(Y_Path[-1].clone())

        # Import PTOutput here to avoid circular import
        from lib.sampler import PTOutput

        return PTOutput(
            samples=samples,
            final_path=Y_Path,
            burn_in_samples=burn_in_samples,
            acceptance_rates=None,  # Could be computed from accept_count_per_timestep if needed
            path_trace=trace_collector.path_trace,
            accept_count_per_timestep=trace_collector.accept_count_per_timestep,
            log_weights_reverse_per_timestep=trace_collector.log_weights_reverse_per_timestep,
            log_weights_forward_per_timestep=trace_collector.log_weights_forward_per_timestep,
            weights_count_per_timestep=trace_collector.weights_count_per_timestep,
            k_paths_storage=trace_collector.k_paths_storage,
        )

    return sample_parallel_tempering
