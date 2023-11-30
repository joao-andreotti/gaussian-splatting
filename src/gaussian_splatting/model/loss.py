__all__ = ["L1", "DSSIM"]
import torch
import torch.nn.functional as F

def L1(
    original_image: torch.Tensor,
    rendered_image: torch.Tensor,
) -> torch.Tensor:
    """Computes the L1 loss between the two images.
    """
    return torch.abs(original_image - rendered_image).mean()

def create_square_window(window_size: int, channels: int) -> torch.Tensor:
    """Creates a square window for averaging."""
    window = torch.ones(
        (channels, 1, window_size, window_size), dtype=torch.float, device="cuda"
    )
    return window / (window_size * window_size)

def DSSIM(
    original_image: torch.Tensor,
    rendered_image: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 1.0
) -> torch.Tensor:
    """Computes the structural dis-similarity index between two images.
    DSSIM = 1 - SSIM
    This formulation adapts SSIM nicely into a loss function.
    Adapted from: Po-Hsun-Su/pytorch-ssim.git
    """
    channels = original_image.shape[0]
    window = create_square_window(window_size, channels=channels)

    mu1 = F.conv2d(original_image, window, padding="same", groups=channels)
    mu2 = F.conv2d(rendered_image, window, padding="same", groups=channels)

    mu1_sq = mu1.square()
    mu2_sq = mu2.square()

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            original_image * original_image, window, padding="same", groups=channels
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            rendered_image * rendered_image, window, padding="same", groups=channels
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            original_image * rendered_image, window, padding="same", groups=channels
        )
        - mu1_mu2
    )

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1 - ssim_map.mean()
