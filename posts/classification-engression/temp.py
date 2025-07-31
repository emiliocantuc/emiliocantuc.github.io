# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

# %% [markdown]
# ## 3-class Classification with Engression-style Sampling
# Learn \( P(Y \mid X) \) using stochastic model and compare with true conditional.

# %%
# Generate synthetic 3-class data
def gen_3class_data(n=5000):
    x = torch.rand(n, 1)
    p0 = 1 - x
    p1 = x * (1 - x)
    p2 = x ** 2
    p = torch.cat([p0, p1, p2], dim=1)
    p = p / p.sum(dim=1, keepdim=True)
    y = torch.multinomial(p, num_samples=1)
    return x, y, p

# %%
x, y, true_p = gen_3class_data()
ds = TensorDataset(x, y, true_p)
dl = DataLoader(ds, batch_size=128, shuffle=True)

# %% [markdown]
# ## Model: MLP + noise injection (engression-style)

# %%
class gConcat(nn.Module):
    def __init__(self, base, m_train=4, m_eval=32, noise_dim=8):
        super().__init__()
        self.base = base
        self.m_train = m_train
        self.m_eval = m_eval
        self.noise_dim = noise_dim

    def forward(self, x):
        m = self.m_train if self.training else self.m_eval
        batch = x.shape[0]
        x_rep = x.unsqueeze(1).expand(batch, m, x.shape[1]).reshape(batch * m, -1)
        noise = torch.rand(batch * m, self.noise_dim)
        x_noisy = torch.cat([x_rep, noise], dim=1)
        out = self.base(x_noisy)
        return out.view(batch, m, -1)

# %%
class SimpleClassifier(nn.Module):
    def __init__(self, noise_dim=8, hidden_dim=32, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)
        )

    def forward(self, x):
        return self.net(x)

# %% [markdown]
# ## Loss and Training

# %%
def train(model, dl, loss_fn, epochs=30, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        for x, y, _ in dl:
            opt.zero_grad()
            preds = model(x)  # shape: (batch, m, k)
            probs = F.log_softmax(preds, dim=-1)
            loss = F.nll_loss(probs.mean(1), y.squeeze())
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses

# %% [markdown]
# ## Evaluation

# %%
@torch.no_grad()
def evaluate(model, dl, noise_dim=8, m=64):
    model.eval()
    all_preds, all_true = [], []

    for x, _, true_p in dl:
        x_rep = x.repeat(m, 1)
        noise = torch.rand_like(x_rep).repeat(1, noise_dim)
        x_noisy = torch.cat([x_rep, noise], dim=1)
        logits = model.base(x_noisy).view(-1, m, 3)
        probs = F.softmax(logits, dim=-1).mean(1)
        all_preds.append(probs)
        all_true.append(true_p)

    pred = torch.cat(all_preds)
    truth = torch.cat(all_true)

    mean_l1 = (pred - truth).abs().mean().item()
    return mean_l1, pred, truth

# %% [markdown]
# ## Train and Evaluate

# %%
noise_dim = 8
k = 3

base = SimpleClassifier(noise_dim=noise_dim, k=k)
model = gConcat(base, m_train=4, m_eval=64, noise_dim=noise_dim)

losses = train(model, dl, F.cross_entropy)
err, pred_probs, true_probs = evaluate(model, dl, noise_dim=noise_dim)

print(f"Mean L1 error: {err:.4f}")

# %% [markdown]
# ## Visualization

# %%
@torch.no_grad()
def plot_preds(model, noise_dim=8, m=64):
    model.eval()
    xs = torch.linspace(0, 1, 200).unsqueeze(1)
    x_rep = xs.repeat(m, 1)
    noise = torch.rand_like(x_rep).repeat(1, noise_dim)
    x_noisy = torch.cat([x_rep, noise], dim=1)
    logits = model.base(x_noisy).view(-1, m, 3)
    probs = F.softmax(logits, dim=-1).mean(1)

    true_p0 = 1 - xs
    true_p1 = xs * (1 - xs)
    true_p2 = xs ** 2
    true_p = torch.cat([true_p0, true_p1, true_p2], dim=1)
    true_p = true_p / true_p.sum(dim=1, keepdim=True)

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(xs.squeeze(), probs[:, i], label=f'Pred class {i}')
        plt.plot(xs.squeeze(), true_p[:, i], '--', label=f'True class {i}')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("P(y|x)")
    plt.title("Predicted vs True Class Probabilities")
    plt.show()

plot_preds(model, noise_dim=noise_dim)

# %%
