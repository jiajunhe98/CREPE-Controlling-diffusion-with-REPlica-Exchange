import abc

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from lib.catsample import sample_categorical


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):
    @property
    def dim(self) -> int:
        """
        dim is the number of states in the graph. It is the size of the vocabulary.
        It is misleading, but don't confuse it with dim the dimension of the data.
        """
        pass

    @property
    def absorb(self) -> bool:
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass

    @abc.abstractmethod
    def rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the i-th column of the rate matrix Q, where i is [batch, datadim]. datadim is the dimension of the data.

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass

    @abc.abstractmethod
    def transp_rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass

    @abc.abstractmethod
    def transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch dim vocab"]:
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass

    def sample_transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Int[torch.Tensor, "batch datadim"]:
        """
        Samples the transition vector.
        """
        transition_vector: Float[torch.Tensor, "batch vocab"] = self.transition(
            i, sigma
        )
        sampled = sample_categorical(transition_vector, method="hard")
        return sampled

    def reverse_rate(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        score: Float[torch.Tensor, "batch datadim vocab"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Constructs the reverse rate. Which is score * transp_rate. Makes sure it is normalised such
        that the sum along the last dimension is 0.
        It is constructed by multiplying the score with the transposed rate matrix.

        This is the rate from i to all other states. For that you want the score evaluated  at i.

        """
        normalized_rate: Float[torch.Tensor, "batch datadim vocab"] = (
            self.transp_rate(i) * score
        )

        # Put zeros along the last dimension of normalized_rate at positions i
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))

        # Normalize the rate so that it sums to 0, by putting the negative sum along the last dimension at positions i
        normalized_rate.scatter_(
            -1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True)
        )
        return normalized_rate

    def sample_rate(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        rate: Float[torch.Tensor, "batch datadim vocab"],
    ) -> tuple[
        Int[torch.Tensor, "batch datadim"], Float[torch.Tensor, "batch datadim 1"]
    ]:
        """
        This construct and samples from p(x_{t + dt} | x_t = i) = delta_{x_{t + dt} = i} + rate_{x_{t + dt} = i}dt

        It is implied that rate = rate_{x_{t + dt} = i}dt
        Returns the sample and the probability of the sample.
        """
        probs: Float[torch.Tensor, "batch datadim vocab"] = (
            F.one_hot(i, num_classes=self.dim).to(rate) + rate
        )
        x: Int[torch.Tensor, "batch datadim"] = sample_categorical(probs)
        probs = torch.gather(probs, -1, x[..., None])
        return x, probs

    @abc.abstractmethod
    def staggered_score(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        dsigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass

    @abc.abstractmethod
    def score_entropy(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        sigma: Float[torch.Tensor, "batch 1"],
        x: Int[torch.Tensor, "batch datadim"],
        x0: Int[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch datadim"]:
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup. Columns sum to 0.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: The dimension of the graph, i.e. the number of states.
        """
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def absorb(self) -> bool:
        return False

    def rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the transition rate from index i to all other indices, batched.
        """
        edge = (
            torch.ones(*i.shape, self.dim, device=i.device, dtype=torch.float32)
            / self.dim
        )
        edge = edge.scatter(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def transp_rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the transition rate of the transposed rate matrix. Uniform graph is symmetric.
        """
        return self.rate(i)

    def transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the transition matrix e^{sigma Q}, going from i to all other indices.
        """
        trans = (
            torch.ones(*i.shape, self.dim, device=i.device)
            * (1 - (-sigma[..., None]).exp())
            / self.dim
        )
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        """
        Computes the transition matrix e^{sigma Q}^T.
        """
        return self.transition(i, sigma)

    def sample_transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Int[torch.Tensor, "batch datadim"]:
        """
        Samples the transition vector p_{t|0}^tok (.|i) i-th column of e^{sigma Q}

        Parameters:
        - i: Index tensor of shape (B,)
        - sigma: Time parameter tensor of shape (B,)

        Returns:
        - Index tensor of the same shape as i
        """

        # Calculate move chance from sigma (shape B,)
        move_chance = 1 - (-sigma).exp()

        # Generate random values matching i's shape - explicitly use float type regardless of i's type
        move_indices = (
            torch.rand_like(i, device=i.device, dtype=torch.float32) < move_chance
        )

        # Generate perturbed indices
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)

        return i_pert

    def staggered_score(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        dsigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(
            dim=-1, keepdim=True
        ) + score / epow

    def sample_limit(self, *batch_dims) -> Int[torch.Tensor, "batch"]:
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        sigma: Float[torch.Tensor, "batch 1"],
        x: Int[torch.Tensor, "batch datadim"],
        x0: Int[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch datadim"]:
        esigm1 = torch.expm1(sigma)
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term: Float[torch.Tensor, "batch datadim"] = (
            score.mean(dim=-1)
            - torch.gather(score, -1, x[..., None].long()).squeeze(-1) / self.dim
        )

        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term: Float[torch.Tensor, "batch datadim"] = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None].long()).squeeze(-1) / esigm1
            + neg_term,
        )

        # constant factor
        const: Float[torch.Tensor, "batch datadim"] = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim,
        )

        # positive term
        sexp = score.exp()
        pos_term: Float[torch.Tensor, "batch datadim"] = (
            sexp.mean(dim=-1)
            - torch.gather(sexp, -1, x[..., None].long()).squeeze(-1) / self.dim
        )

        result = pos_term - neg_term + const
        return result


class Absorbing(Graph):
    def __init__(self, dim: int, debug: bool = False):
        super().__init__()
        self._dim = dim
        self.debug = debug

    @property
    def dim(self) -> int:
        return self._dim + 1

    @property
    def absorb(self) -> bool:
        return True

    def rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return (
            F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim)
            - F.one_hot(i, num_classes=self.dim)
        ).float()

    def transp_rate(
        self, i: Int[torch.Tensor, "batch datadim"]
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch dim vocab"]:
        """
        Computes the transition matrix e^{sigma Q}, going from i to all other indices.
        e^{sigma Q} = I - (e^{-sigma}-1)Q
        """
        return F.one_hot(i, num_classes=self.dim) - (
            (-sigma).exp() - torch.ones_like(sigma)
        )[..., None] * self.rate(i)

    def transp_transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(i == self.dim - 1, 1 - (-sigma).squeeze(-1).exp(), 0)[
            ..., None
        ]
        return edge

    def sample_transition(
        self,
        i: Int[torch.Tensor, "batch datadim"],
        sigma: Float[torch.Tensor, "batch 1"],
    ) -> Int[torch.Tensor, "batch datadim"]:
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        dsigma: Float[torch.Tensor, "batch 1"],
    ) -> Float[torch.Tensor, "batch datadim vocab"]:
        score = score.clone()  # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims) -> Int[torch.Tensor, "batch datadim"]:
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def _score_entropy(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        sigma: Float[torch.Tensor, "batch 1"],
        x: Int[torch.Tensor, "batch datadim"],
        x0: Int[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch datadim"]:
        """
        Computes the score entropy function for the Absorbing graph model without debugging output.

        Args:
            score: Log scores from model with shape [batch, datadim, vocab]
            sigma: Noise levels with shape [batch, 1], one per batch element
            x: Current noisy tokens with shape [batch, datadim]
            x0: Original tokens with shape [batch, datadim]

        Returns:
            Score entropy with shape [batch, datadim]
        """
        rel_ind = x == self.dim - 1

        # Calculate expm1 with the numerical trick for better stability
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        entropy = torch.zeros(*x.shape, device=x.device)

        # Compute the ratio 1/esigm1 for corrupted positions
        ratio = 1 / esigm1.expand_as(x)[rel_ind]

        # Get original token indices for corrupted positions
        other_ind = x0[rel_ind]

        # Negative term: scale the original token scores
        gathered_scores = torch.gather(score[rel_ind], -1, other_ind[..., None].long())
        neg_term = ratio * gathered_scores.squeeze(-1)

        # Positive term: sum exp of scores for all non-absorbing states
        scores_before_exp = score[rel_ind][:, :-1]
        scores_exp = scores_before_exp.exp()
        pos_term = scores_exp.sum(dim=-1)

        # Constant term: ratio * (log(ratio) - 1)
        ratio_log = ratio.log()
        const = ratio * (ratio_log - 1)

        # Combine terms for final entropy
        entropy[rel_ind] = pos_term - neg_term + const

        return entropy

    def _debug_score_entropy(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        sigma: Float[torch.Tensor, "batch 1"],
        x: Int[torch.Tensor, "batch datadim"],
        x0: Int[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch datadim"]:
        """
        Computes the score entropy function for the Absorbing graph model.

        This function calculates entropy terms specifically for elements that have been
        corrupted to the absorbing state (x == dim-1). Each element in the batch is
        processed independently with its own noise level.

        The calculation consists of three components:
        1. Positive term: Sum of exponential scores for all non-absorbing states
        2. Negative term: Scaled score at the original data position
        3. Constant term: A stabilizing factor based on noise level

        The implementation uses a numerical trick (torch.where) to calculate expm1
        more accurately for small sigma values.

        Args:
            score: Log scores from model with shape [batch, datadim, vocab]
            sigma: Noise levels with shape [batch, 1], one per batch element
            x: Current noisy tokens with shape [batch, datadim]
            x0: Original tokens with shape [batch, datadim]

        Returns:
            Score entropy with shape [batch, datadim], where each element depends
            only on its own noise level (no mixing between batch elements)
        """
        print("\n===== SCORE_ENTROPY DEBUG =====")
        print(
            f"Input score shape: {score.shape}, min: {score.min().item():.6f}, max: {score.max().item():.6f}"
        )
        print(
            f"Input sigma shape: {sigma.shape}, min: {sigma.min().item():.6f}, max: {sigma.max().item():.6f}"
        )
        print(
            f"Is sigma inf: {torch.isinf(sigma).any().item()}, Is sigma nan: {torch.isnan(sigma).any().item()}"
        )

        rel_ind = x == self.dim - 1
        print(
            f"rel_ind shape: {rel_ind.shape}, sum: {rel_ind.sum().item()} (number of absorbing states)"
        )

        # Debug before esigm1 calculation
        print(f"sigma < 0.5 count: {(sigma < 0.5).sum().item()}")

        # Calculate esigm1 with extra safety
        try:
            esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
            print(
                f"esigm1 shape: {esigm1.shape}, min: {esigm1.min().item():.6f}, max: {esigm1.max().item():.6f}"
            )
            print(
                f"Is esigm1 inf: {torch.isinf(esigm1).any().item()}, Is esigm1 nan: {torch.isnan(esigm1).any().item()}"
            )
            if torch.isinf(esigm1).any():
                inf_indices = torch.where(torch.isinf(esigm1))
                print(f"Inf esigm1 at indices: {inf_indices}")
                print(f"Corresponding sigma values: {sigma[inf_indices]}")
        except Exception as e:
            print(f"Exception in esigm1 calculation: {e}")

        # Debug ratio calculation - division by very small values can cause inf
        try:
            esigm1_expanded = esigm1.expand_as(x)
            print(
                f"esigm1_expanded[rel_ind] min: {esigm1_expanded[rel_ind].min().item() if rel_ind.sum() > 0 else 'N/A'}"
            )

            ratio = 1 / esigm1.expand_as(x)[rel_ind]
            print(
                f"ratio shape: {ratio.shape}, min: {ratio.min().item() if ratio.numel() > 0 else 'N/A':.6f}, "
                f"max: {ratio.max().item() if ratio.numel() > 0 else 'N/A':.6f}"
            )
            print(
                f"Is ratio inf: {torch.isinf(ratio).any().item() if ratio.numel() > 0 else 'N/A'}, "
                f"Is ratio nan: {torch.isnan(ratio).any().item() if ratio.numel() > 0 else 'N/A'}"
            )
            if torch.isinf(ratio).any():
                inf_indices = torch.where(torch.isinf(ratio))
                print(f"Inf ratio at indices: {inf_indices}")
        except Exception as e:
            print(f"Exception in ratio calculation: {e}")

        other_ind = x0[rel_ind]
        print(f"other_ind shape: {other_ind.shape}")

        # Debug negative term calculation
        try:
            gathered_scores = torch.gather(
                score[rel_ind], -1, other_ind[..., None].long()
            )
            print(
                f"gathered_scores shape: {gathered_scores.shape}, "
                f"min: {gathered_scores.min().item() if gathered_scores.numel() > 0 else 'N/A'}, "
                f"max: {gathered_scores.max().item() if gathered_scores.numel() > 0 else 'N/A'}"
            )

            neg_term = ratio * gathered_scores.squeeze(-1)
            print(
                f"neg_term shape: {neg_term.shape}, "
                f"min: {neg_term.min().item() if neg_term.numel() > 0 else 'N/A'}, "
                f"max: {neg_term.max().item() if neg_term.numel() > 0 else 'N/A'}"
            )
            print(
                f"Is neg_term inf: {torch.isinf(neg_term).any().item() if neg_term.numel() > 0 else 'N/A'}, "
                f"Is neg_term nan: {torch.isnan(neg_term).any().item() if neg_term.numel() > 0 else 'N/A'}"
            )
        except Exception as e:
            print(f"Exception in negative term calculation: {e}")

        # Debug positive term calculation - exp can overflow
        try:
            scores_before_exp = score[rel_ind][:, :-1]
            print(
                f"scores_before_exp shape: {scores_before_exp.shape}, "
                f"min: {scores_before_exp.min().item() if scores_before_exp.numel() > 0 else 'N/A'}, "
                f"max: {scores_before_exp.max().item() if scores_before_exp.numel() > 0 else 'N/A'}"
            )

            # Check for extremely large values before exp
            large_scores = (
                scores_before_exp > 50
            )  # exp(50) is already a very large number
            if large_scores.any():
                print(
                    f"WARNING: Found {large_scores.sum().item()} values > 50 before exp, which could cause overflow"
                )

            scores_exp = scores_before_exp.exp()
            print(
                f"scores_exp shape: {scores_exp.shape}, "
                f"min: {scores_exp.min().item() if scores_exp.numel() > 0 else 'N/A'}, "
                f"max: {scores_exp.max().item() if scores_exp.numel() > 0 else 'N/A'}"
            )
            print(
                f"Is scores_exp inf: {torch.isinf(scores_exp).any().item() if scores_exp.numel() > 0 else 'N/A'}, "
                f"Is scores_exp nan: {torch.isnan(scores_exp).any().item() if scores_exp.numel() > 0 else 'N/A'}"
            )

            pos_term = scores_exp.sum(dim=-1)
            print(
                f"pos_term shape: {pos_term.shape}, "
                f"min: {pos_term.min().item() if pos_term.numel() > 0 else 'N/A'}, "
                f"max: {pos_term.max().item() if pos_term.numel() > 0 else 'N/A'}"
            )
            print(
                f"Is pos_term inf: {torch.isinf(pos_term).any().item() if pos_term.numel() > 0 else 'N/A'}, "
                f"Is pos_term nan: {torch.isnan(pos_term).any().item() if pos_term.numel() > 0 else 'N/A'}"
            )
        except Exception as e:
            print(f"Exception in positive term calculation: {e}")

        # Debug constant term calculation - log of very small values can be problematic
        try:
            ratio_log = ratio.log()
            print(
                f"ratio_log shape: {ratio_log.shape}, "
                f"min: {ratio_log.min().item() if ratio_log.numel() > 0 else 'N/A'}, "
                f"max: {ratio_log.max().item() if ratio_log.numel() > 0 else 'N/A'}"
            )
            print(
                f"Is ratio_log inf: {torch.isinf(ratio_log).any().item() if ratio_log.numel() > 0 else 'N/A'}, "
                f"Is ratio_log nan: {torch.isnan(ratio_log).any().item() if ratio_log.numel() > 0 else 'N/A'}"
            )

            const = ratio * (ratio_log - 1)
            print(
                f"const shape: {const.shape}, "
                f"min: {const.min().item() if const.numel() > 0 else 'N/A'}, "
                f"max: {const.max().item() if const.numel() > 0 else 'N/A'}"
            )
            print(
                f"Is const inf: {torch.isinf(const).any().item() if const.numel() > 0 else 'N/A'}, "
                f"Is const nan: {torch.isnan(const).any().item() if const.numel() > 0 else 'N/A'}"
            )
        except Exception as e:
            print(f"Exception in constant term calculation: {e}")

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const

        # Debug final entropy
        print(
            f"Final entropy shape: {entropy.shape}, "
            f"min: {entropy.min().item()}, max: {entropy.max().item()}"
        )
        print(
            f"Is entropy inf: {torch.isinf(entropy).any().item()}, "
            f"Is entropy nan: {torch.isnan(entropy).any().item()}"
        )
        if torch.isinf(entropy).any():
            inf_indices = torch.where(torch.isinf(entropy))
            print(f"Infinite values found at indices: {inf_indices}")
            print(f"Components at these positions:")
            if rel_ind[inf_indices].any():
                rel_idx = torch.where(rel_ind[inf_indices])
                if rel_idx[0].numel() > 0:
                    pos_inf = pos_term[rel_idx[0]]
                    neg_inf = neg_term[rel_idx[0]]
                    const_inf = const[rel_idx[0]]
                    print(
                        f"  pos_term: {pos_inf.item() if pos_inf.numel() == 1 else pos_inf}"
                    )
                    print(
                        f"  neg_term: {neg_inf.item() if neg_inf.numel() == 1 else neg_inf}"
                    )
                    print(
                        f"  const: {const_inf.item() if const_inf.numel() == 1 else const_inf}"
                    )
        print("===== END SCORE_ENTROPY DEBUG =====\n")

        return entropy

    def score_entropy(
        self,
        score: Float[torch.Tensor, "batch datadim vocab"],
        sigma: Float[torch.Tensor, "batch 1"],
        x: Int[torch.Tensor, "batch datadim"],
        x0: Int[torch.Tensor, "batch datadim"],
    ) -> Float[torch.Tensor, "batch datadim"]:
        """
        Computes the score entropy function for the Absorbing graph model.

        Delegates to either the debug version or the standard version based on the debug flag.

        Args:
            score: Log scores from model with shape [batch, datadim, vocab]
            sigma: Noise levels with shape [batch, 1], one per batch element
            x: Current noisy tokens with shape [batch, datadim]
            x0: Original tokens with shape [batch, datadim]

        Returns:
            Score entropy with shape [batch, datadim]
        """
        if self.debug:
            return self._debug_score_entropy(score, sigma, x, x0)
        else:
            return self._score_entropy(score, sigma, x, x0)
