import torch
import torch.nn as nn
from tqdm.auto import tqdm
import gc

from cpflow.flows import SequentialFlow, ActNorm
from cpflow.cpflows import DeepConvexFlow
from cpflow.icnn import PICNN
from protocols.pushforward_operator import PushForwardOperator
from utils import TrainParams


class CPFlow(PushForwardOperator, nn.Module):
    def __init__(
        self,
        response_dimension: int,
        feature_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.config = dict(
            response_dimension=response_dimension,
            feature_dimension=feature_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            n_blocks=n_blocks,
        )

        icnns = [
            PICNN(
                dim=response_dimension,
                dimh=hidden_dimension,
                dimc=feature_dimension,
                num_hidden_layers=number_of_hidden_layers,
                symm_act_first=True,
                softplus_type="gaussian_softplus",
                zero_softplus=True,
            )
            for _ in range(n_blocks)
        ]
        layers = [None] * (2 * n_blocks + 1)
        layers[0::2] = [ActNorm(response_dimension) for _ in range(n_blocks + 1)]
        layers[1::2] = [
            DeepConvexFlow(icnn, response_dimension, unbiased=False)
            for _, icnn in zip(range(n_blocks), icnns)
        ]
        self.flow = SequentialFlow(layers)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        train_params: TrainParams,
        *args,
        **kwargs,
    ):
        num_epochs = train_params.get("num_epochs", 100)
        lr = train_params.get("learning_rate", 1e-3)
        verbose = train_params.get("verbose", True)
        print_every = None
        if verbose:
            print_every = 10

        optim = torch.optim.Adam(self.flow.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, num_epochs * len(train_loader), eta_min=0
        )

        loss_acc = 0
        t = 0

        self.flow.train()
        for _ in tqdm(range(num_epochs)):
            for cond, y in train_loader:
                loss = -self.flow.logp(y, context=cond).mean()
                optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.flow.parameters(), max_norm=10
                ).item()

                optim.step()
                sch.step()

                loss_acc += loss.item()
                del loss
                gc.collect()
                torch.clear_autocast_cache()

                t += 1
                if t == 1:
                    print("init loss:", loss_acc)
                if print_every is not None and t % print_every == 0:
                    print(t, loss_acc / print_every)
        self.is_fitted_ = True
        return self

    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            y = self.flow.reverse(U, context=X)

        return y

    def push_backward_y_given_x(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            u, _ = self.flow.forward_transform(Y, context=X)
        return u

    def sample_y_given_x(self, n_samples: int, X: torch.Tensor) -> torch.Tensor:
        u = torch.randn(
            n_samples,
            self.config["response_dimension"],
            device=X.device,
            dtype=torch.float32,
        )
        y = self.push_forward_u_given_x(u, X)
        return y

    def logp_cond(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted_:
            raise ValueError(
                "This CPFlow instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.flow.logp(Y, context=X)

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save(
            {
                "state_dict": self.flow.state_dict(),
                **self.config,
            },
            path,
        )

    def load(self, path: str):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path)
        with torch.no_grad():
            y = torch.rand(8, self.config["response_dimension"])
            x = torch.rand(8, self.config["feature_dimension"])
        self.flow.forward_transform(y, context=x)
        self.flow.load_state_dict(data["state_dict"])
        self.config.update(data)
        self.is_fitted_ = True
        return self
