"""Deep Fingerprinting CNN (Sirinam et al., CCS 2018).

Four convolutional blocks over the raw packet-direction sequence, then two
fully connected layers. Hyperparameters follow Table 5 of the paper; the only
deviation is `input_length`, which tracks our capture format rather than the
paper's 5000 Tor cells.
"""

import numpy as np
import torch
import torch.nn as nn

FILTER_SIZES = [32, 64, 128, 256]
KERNEL_SIZE = 8
POOL_SIZE = 8
POOL_STRIDE = 4
CONV_DROPOUT = 0.1


def _block(in_ch, out_ch, activation):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, KERNEL_SIZE, stride=1, padding="same"),
        nn.BatchNorm1d(out_ch),
        activation(),
        nn.Conv1d(out_ch, out_ch, KERNEL_SIZE, stride=1, padding="same"),
        nn.BatchNorm1d(out_ch),
        activation(),
        nn.MaxPool1d(POOL_SIZE, stride=POOL_STRIDE, padding=POOL_SIZE // 2),
        nn.Dropout(CONV_DROPOUT),
    )


class DFNet(nn.Module):
    def __init__(self, n_classes, input_length=5000):
        super().__init__()
        # The paper uses ELU in the first block only, ReLU in the rest.
        activations = [nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU]
        channels = [1] + FILTER_SIZES

        self.features = nn.Sequential(
            *[
                _block(channels[i], channels[i + 1], activations[i])
                for i in range(len(FILTER_SIZES))
            ]
        )

        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, input_length)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_df(
    X_train,
    y_train,
    n_classes,
    epochs=30,
    batch_size=128,
    lr=0.002,
    seed=42,
    device="cpu",
    verbose=True,
    X_val=None,
    y_val=None,
    patience=20,
):
    """Train DF; with a validation set, stop early and keep the best epoch.

    The paper's fixed 30 epochs assumes 800+ traces per class. At ~30 per
    class the model is still underfitting at 30 and overfitting by 100, so a
    fixed count is wrong in both directions. Returns (model, best_epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DFNet(n_classes, input_length=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.from_numpy(X_train).unsqueeze(1)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X) > batch_size,  # BatchNorm needs >1 sample per batch
    )

    use_val = X_val is not None and len(X_val) > 0
    best_acc, best_epoch, best_state = -1.0, epochs, None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)

        if verbose and (epoch % 5 == 4 or epoch == epochs - 1):
            print(
                f"    epoch {epoch + 1:>2}/{epochs}  "
                f"loss {total_loss / max(seen, 1):.4f}  "
                f"train acc {correct / max(seen, 1):.4f}",
                flush=True,
            )

        if use_val:
            val_acc = float(
                (predict_proba(model, X_val, device=device).argmax(1) == y_val).mean()
            )
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch + 1 - best_epoch >= patience:
                if verbose:
                    print(
                        f"    early stop at epoch {epoch + 1}, "
                        f"best val acc {best_acc:.4f} at epoch {best_epoch}",
                        flush=True,
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def predict_proba(model, X, batch_size=256, device="cpu"):
    model.eval()
    out = []
    X = torch.from_numpy(X).unsqueeze(1)
    for i in range(0, len(X), batch_size):
        logits = model(X[i : i + batch_size].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)
