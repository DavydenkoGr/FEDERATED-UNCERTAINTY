import sys
import torch
import numpy as np
from tqdm.auto import tqdm
import gc

sys.path.insert(0, "./third_party/cp-flow")
from lib.flows import SequentialFlow, DeepConvexFlow, ActNorm
from lib.icnn import ICNN3


class CPFlowOrdering:
    def __init__(
        self,
        dimx: int,
        hidden_dim: int,
        num_hidden_layers: int,
        nblocks: int,
        zero_softplus: bool = True,
        softplus_type: str = "gaussian_softplus",
        symm_act_first: bool = True,
        device: str = "cuda:0",
    ):
        self.device = device
        self.dimx = dimx
        icnns = [
            ICNN3(
                dimx,
                hidden_dim,
                num_hidden_layers,
                symm_act_first=symm_act_first,
                softplus_type=softplus_type,
                zero_softplus=zero_softplus,
            )
            for _ in range(nblocks)
        ]
        if nblocks == 1:
            # for printing the potential only
            layers = [None] * (nblocks + 1)
            # noinspection PyTypeChecker
            layers[0] = ActNorm(dimx)
            layers[1:] = [
                DeepConvexFlow(icnn, dimx, unbiased=False, bias_w1=-0.0)
                for _, icnn in zip(range(nblocks), icnns)
            ]
        else:
            layers = [None] * (2 * nblocks + 1)
            layers[0::2] = [ActNorm(dimx) for _ in range(nblocks + 1)]
            layers[1::2] = [
                DeepConvexFlow(
                    icnn, dimx, unbiased=False, bias_w1=-0.0, trainable_w0=False
                )
                for _, icnn in zip(range(nblocks), icnns)
            ]
        self.flow = SequentialFlow(layers)

    def fit(self, scores_cal: np.ndarray, train_params: dict):
        num_epochs = train_params.get("num_epochs", 100)
        batch_size = train_params.get("batch_size", 128)
        lr = train_params.get("lr", 1e-3)
        print_every = train_params.get("print_every", 10)

        self.flow = self.flow.to(self.device)

        train_loader = torch.utils.data.DataLoader(
            torch.tensor(scores_cal, dtype=torch.float32, device=self.device),
            batch_size=batch_size,
            shuffle=True,
        )

        optim = torch.optim.AdamW(self.flow.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, num_epochs * len(train_loader), eta_min=0
        )

        loss_acc = 0
        t = 0

        self.flow.train()
        for _ in tqdm(range(num_epochs)):
            for x in train_loader:
                x = x.view(-1, self.dimx)
                x = x.to(self.device)

                loss = -self.flow.logp(x).mean()
                optim.zero_grad()
                loss.backward()

                optim.step()
                sch.step()

                loss_acc += loss.item()
                del loss
                gc.collect()
                torch.clear_autocast_cache()

                t += 1
                if t == 1:
                    print("init loss:", loss_acc)
                if t % print_every == 0:
                    print(t, loss_acc / print_every)
        self.is_fitted_ = True
        return self

    def predict(self, scores_test: np.ndarray):
        with torch.no_grad():
            self.flow.eval()
            for f in self.flow.flows[1::2]:
                f.no_bruteforce = False
            z_, _ = self.flow.forward_transform(
                torch.tensor(scores_test, dtype=torch.float32, device=self.device),
                context=None,
            )
            pushforward_of_u = z_.cpu()

        mk_norms = np.linalg.norm(pushforward_of_u, axis=-1, ord=2)
        ordering_indices = np.argsort(mk_norms)

        return mk_norms, ordering_indices

    def predict_ranks(self, scores_test):
        if not self.is_fitted_:
            raise ValueError(
                "This OTCPOrdering instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        mk_norms, _ = self.predict(scores_test)

        return mk_norms
