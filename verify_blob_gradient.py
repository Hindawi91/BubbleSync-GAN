"""
Empirical test: does a gradient actually reach x_fake through
get_blobs_properties() + MSELoss, the same way solver.py's original
(unfixed) code would have used it?

Run with the same Python environment used for training:
    python3 verify_blob_gradient.py

Expected output on the ORIGINAL (pre-fix) get_blobs_properties():
    blob_counts_fake.requires_grad: False
    blob_counts_fake.grad_fn: None
    loss.requires_grad: False
    backward() failed: element 0 of tensors does not require grad and does not have a grad_fn

This confirms the bug fixed by get_blobs_properties_differentiable.py (see
that file's docstring for the full explanation): blob statistics computed
via skimage/NumPy have no path back to the Generator's computation graph,
so the count/mean-area/std-area losses contributed zero gradient
regardless of their lambda weights.
"""
import torch
import torch.nn as nn
from get_blobs_properties import get_blobs_properties

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulate x_fake: pretend this came from the Generator (requires_grad=True,
# same as real Generator output would have).
x_fake = torch.rand(2, 1, 64, 64, device=device, requires_grad=True)
x_real = torch.rand(2, 1, 64, 64, device=device)  # ground truth, no grad needed
labels = torch.tensor([0, 1])

blob_counts_real, blob_mean_real, blob_std_real = get_blobs_properties(
    images=x_real, labels=labels, device=device, source_domain='DS3', target_domain='DS1'
)
blob_counts_fake, blob_mean_fake, blob_std_fake = get_blobs_properties(
    images=x_fake, labels=labels, device=device, source_domain='DS3', target_domain='DS1'
)

print("blob_counts_fake.requires_grad:", blob_counts_fake.requires_grad)
print("blob_counts_fake.grad_fn:", blob_counts_fake.grad_fn)

loss = nn.MSELoss()(blob_counts_real, blob_counts_fake)
print("loss.requires_grad:", loss.requires_grad)

try:
    loss.backward()
    print("x_fake.grad is None:", x_fake.grad is None)
    if x_fake.grad is not None:
        print("x_fake.grad nonzero elements:", (x_fake.grad != 0).sum().item(), "/", x_fake.grad.numel())
except RuntimeError as e:
    print("backward() failed:", e)
