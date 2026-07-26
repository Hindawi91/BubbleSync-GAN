"""
Differentiable gradient routing for blob property extraction (count, mean
area, std area), for use ONLY on the generator's output (the "fake" image
side of the blob losses in solver.py).

Background / why this exists:
The original get_blobs_properties() (get_blobs_properties.py) computes blob
statistics via skimage (Gaussian blur, Otsu thresholding, connected-component
labeling). These are plain NumPy/skimage operations, not PyTorch ops. The
returned tensors are built fresh via torch.tensor(...), which have no
grad_fn -- they are leaf nodes completely disconnected from the Generator's
computation graph. As a result, in solver.py's original code, the blob count
/ mean area / std area losses contributed EXACTLY ZERO gradient to the
Generator regardless of their lambda weights, making lambda_count,
lambda_mean, and lambda_std have no effect on training whatsoever (confirmed
empirically: independent experiments varying only these lambdas produced
bit-for-bit identical trained models; see verify_blob_gradient.py).

Design (straight-through estimator, value-faithful version):
  - The FORWARD pass calls the ORIGINAL, UNMODIFIED get_blobs_properties()
    directly -- so the actual reported count/mean-area/std-area values are
    byte-identical to what the existing (unmodified) code has always
    computed. No reimplementation of Gaussian blur / thresholding /
    small-object removal / connected-component labeling; that logic is
    untouched and lives only in get_blobs_properties.py.
  - The BACKWARD pass needs *some* differentiable path back to the input
    image, since the forward computation above has none. We build a
    separate, lightweight differentiable "soft mask" (Conv2d Gaussian blur +
    sigmoid soft-threshold around the same Otsu value) purely as a gradient
    vehicle -- it is NEVER used to determine the reported values, only to
    give the optimizer a smooth, directionally-correct signal (more/bigger
    soft-mask area -> more/bigger blobs) for how to adjust the image to
    move the count/area statistics in the right direction.

This means: the LOSS VALUES you see in logs are exactly as trustworthy as
the original code's (same algorithm, same numbers). The GRADIENT is an
approximation (as any solution to a fundamentally non-differentiable
problem must be) -- treat it as a heuristic descent direction, not an exact
derivative. std_area's proxy (local windowed variance of the soft mask) is
the weakest of the three; treat lambda_std's gradient with more skepticism
than lambda_count/lambda_mean.

This has NOT been empirically validated beyond confirming gradients are
non-zero, finite, and that varying lambda_count now actually produces
different trained models (see the smoke test comparing exp3/exp5 in
conversation). Continue monitoring training stability during full runs.
"""
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
import numpy as np
from get_blobs_properties import get_blobs_properties


def _gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d, radius


def _differentiable_gaussian_blur(images, sigma):
    """images: (B, 1, H, W) tensor, differentiable."""
    kernel, radius = _gaussian_kernel(sigma, images.device, images.dtype)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    return F.conv2d(images, kernel, padding=radius)


def _build_gradient_vehicle_soft_mask(images, sigma=1.5, temperature=0.02):
    """Builds a differentiable soft mask used ONLY to route gradients in the
    backward pass below -- never used to compute the reported blob values.
    """
    blurred = _differentiable_gaussian_blur(images, sigma)

    with torch.no_grad():
        blurred_np = blurred.detach().cpu().numpy()
        thresholds = []
        for b in range(blurred_np.shape[0]):
            try:
                t = threshold_otsu(blurred_np[b, 0])
            except ValueError:
                t = float(np.mean(blurred_np[b, 0]))
            thresholds.append(t)
        thresh_t = torch.tensor(thresholds, dtype=images.dtype, device=images.device).view(-1, 1, 1, 1)

    soft_mask = torch.sigmoid((thresh_t - blurred) / temperature)
    return soft_mask


class _BlobStatsSTE(torch.autograd.Function):
    """Straight-through estimator.

    Forward: calls the ORIGINAL get_blobs_properties() on the actual images
    -- exact, unmodified values, no approximation.

    Backward: substitutes a gradient computed from a separately-built
    differentiable soft mask (passed in as `soft_mask`, itself a
    differentiable function of `images`). Autograd routes the returned
    gradient through soft_mask's own graph back to `images` and onward to
    the Generator -- `images` itself receives no direct gradient from this
    Function (it doesn't need one; the numpy-based forward computation has
    no graph to attach one to).
    """

    @staticmethod
    def forward(ctx, images, soft_mask, labels, device, source_domain, target_domain):
        counts, mean_areas, std_areas = get_blobs_properties(
            images=images, labels=labels, device=device,
            source_domain=source_domain, target_domain=target_domain
        )
        ctx.save_for_backward(soft_mask)
        return counts, mean_areas, std_areas

    @staticmethod
    def backward(ctx, grad_count, grad_mean_area, grad_std_area):
        soft_mask, = ctx.saved_tensors
        B = soft_mask.shape[0]

        # Proxy for count/mean_area: total soft-mask area per image.
        area_grad = (grad_count + grad_mean_area).view(B, 1, 1, 1).expand_as(soft_mask)

        # Weakest proxy: local windowed variance of the soft mask as a stand-in
        # for "spread of blob sizes" (std area).
        local_mean = F.avg_pool2d(soft_mask, kernel_size=9, stride=1, padding=4)
        local_sq_mean = F.avg_pool2d(soft_mask ** 2, kernel_size=9, stride=1, padding=4)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        std_grad = grad_std_area.view(B, 1, 1, 1) * local_var

        grad_soft_mask = area_grad + std_grad

        # Gradient order must match forward's positional args:
        # (images, soft_mask, labels, device, source_domain, target_domain)
        return None, grad_soft_mask, None, None, None, None


def get_blobs_properties_differentiable(images, labels, device, source_domain, target_domain,
                                         sigma=1.5, temperature=0.02):
    """Drop-in replacement for get_blobs_properties(), for use ONLY on
    generator output (fake images) where gradients need to flow back to G.

    Reported values are IDENTICAL to calling get_blobs_properties() directly
    (same underlying function, same computation). The only difference is
    that gradients now flow (approximately) back to `images` via a separate
    differentiable soft mask, instead of being silently zero.
    """
    soft_mask = _build_gradient_vehicle_soft_mask(images, sigma=sigma, temperature=temperature)
    return _BlobStatsSTE.apply(images, soft_mask, labels, device, source_domain, target_domain)


if __name__ == "__main__":
    # Sanity check: (1) values match get_blobs_properties() exactly,
    # (2) gradients are non-zero and finite.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    x = torch.rand(2, 1, 64, 64, device=device, requires_grad=True)
    labels = torch.tensor([0, 1])

    # Value-fidelity check against the original function.
    with torch.no_grad():
        ref_counts, ref_mean, ref_std = get_blobs_properties(
            images=x.detach(), labels=labels, device=device,
            source_domain='DS3', target_domain='DS1'
        )

    counts, mean_areas, std_areas = get_blobs_properties_differentiable(
        x, labels, device, source_domain='DS3', target_domain='DS1'
    )
    print("counts match original exactly:", torch.equal(counts, ref_counts))
    print("mean_areas match original exactly:", torch.equal(mean_areas, ref_mean))
    print("std_areas match original exactly:", torch.equal(std_areas, ref_std))

    loss = counts.sum() + mean_areas.sum() + std_areas.sum()
    loss.backward()

    print("grad is None:", x.grad is None)
    if x.grad is not None:
        print("grad nonzero elements:", (x.grad != 0).sum().item(), "/", x.grad.numel())
        print("grad finite:", torch.isfinite(x.grad).all().item())
