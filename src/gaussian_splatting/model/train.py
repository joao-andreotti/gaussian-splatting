__all__ = ["train"]
from typing import Callable
import random

import torch


from gaussian_splatting.model.view import View
from gaussian_splatting.model.gaussian_cloud import GaussianCloud
from gaussian_splatting.model.util import position_points, create_rasterizer
from gaussian_splatting.model.loss import L1, DSSIM


def adaptive_density_control(
    gaussian_cloud: GaussianCloud,
    *,
    opacity_threshold: float = 5e-3,
    positional_threshold: float = 2e-4,
    reconstruction_threshold: float = 5e-2,
    over_reconstruction_scaling_factor: float = 1.6
) -> GaussianCloud:
    """Apply the adaptive density control algorithm (ADC).
    DOES NOT update the gaussian cloud passed as parameter.

    Args:
        gaussian_cloud (const): input gaussian cloud in which to perform the ADC.
            Make sure that the gaussian cloud parameters are all in the same device.
        opacity_threshold: minimum opacity to keep a gaussian in the cloud.
        positional_threshold (tau_pos): minimum gradient in view-space (means2D) to consider for 
            reconstruction.
        reconstruction_threshold (|S|): scale (L2 norm of the gaussian scale vector) threshold to separate
            under-reconstructed from over-reconstructed gaussians.
        over_reconstruction_scaling_factor (phi): factor to divide scales by on over-reconstruction.

    Returns:
        New gaussian cloud with the adapted parameters in the same device as the old one.  
    """

    almost_transparent = (gaussian_cloud.opacities < opacity_threshold).squeeze()
    needs_adapting = gaussian_cloud.means2D.grad.square().sum(dim=1).sqrt() > positional_threshold
    under_reconstructed = needs_adapting & (gaussian_cloud.scales.square().sum(dim=1).sqrt() < reconstruction_threshold)
    over_reconstructed = needs_adapting & (gaussian_cloud.scales.square().sum(dim=1).sqrt() >= reconstruction_threshold)

    filter_points = (
        almost_transparent|
        needs_adapting
    )

    # lets double gaussians that are under_reconstructed
    under_reconstructed_means3D = torch.vstack([
        gaussian_cloud.means3D[under_reconstructed], 
        gaussian_cloud.means3D[under_reconstructed] + gaussian_cloud.means3D.grad[under_reconstructed]
    ])
    under_reconstructed_means2D = torch.vstack([gaussian_cloud.means2D[under_reconstructed]] * 2)
    under_reconstructed_scales = torch.vstack([gaussian_cloud.scales[under_reconstructed]] * 2)
    under_reconstructed_opacities = torch.vstack([gaussian_cloud.opacities[under_reconstructed]] * 2)
    under_reconstructed_shs = torch.vstack([gaussian_cloud.shs[under_reconstructed]] * 2)
    under_reconstructed_rotations = torch.vstack([gaussian_cloud.rotations[under_reconstructed]] * 2)

    # lets double gaussians that are very over_reconstructed

    # sample from gaussian (0, 1 distribution) 
    first_set_points = torch.normal(0, 1, size=(over_reconstructed.sum(), 3)).cuda()
    first_rotated = position_points(gaussian_cloud.scales[over_reconstructed], gaussian_cloud.rotations[over_reconstructed], first_set_points)

    second_set_points = torch.normal(0, 1, size=(over_reconstructed.sum(), 3)).cuda()
    second_rotated = position_points(gaussian_cloud.scales[over_reconstructed], gaussian_cloud.rotations[over_reconstructed], second_set_points)
    
    over_reconstructed_means3D = torch.vstack([
        gaussian_cloud.means3D[over_reconstructed] + first_rotated, 
        gaussian_cloud.means3D[over_reconstructed] + second_rotated
    ])
    over_reconstructed_means2D = torch.vstack([gaussian_cloud.means2D[over_reconstructed]] * 2)
    over_reconstructed_scales = torch.vstack([gaussian_cloud.scales[over_reconstructed]] * 2) / over_reconstruction_scaling_factor
    over_reconstructed_opacities = torch.vstack([gaussian_cloud.opacities[over_reconstructed]] * 2)
    over_reconstructed_shs = torch.vstack([gaussian_cloud.shs[over_reconstructed]] * 2)
    over_reconstructed_rotations = torch.vstack([gaussian_cloud.rotations[over_reconstructed]] * 2)

    # stacking everything
    new_means3D = torch.vstack([    gaussian_cloud.means3D[~filter_points],        over_reconstructed_means3D,     under_reconstructed_means3D])
    new_means2D = torch.vstack([    gaussian_cloud.means2D[~filter_points],        over_reconstructed_means2D,     under_reconstructed_means2D])
    new_scales = torch.vstack([     gaussian_cloud.scales[~filter_points],         over_reconstructed_scales,      under_reconstructed_scales])
    new_opacities = torch.vstack([  gaussian_cloud.opacities[~filter_points],      over_reconstructed_opacities,   under_reconstructed_opacities])
    new_shs = torch.vstack([        gaussian_cloud.shs[~filter_points],            over_reconstructed_shs,         under_reconstructed_shs])
    new_rotations = torch.vstack([  gaussian_cloud.rotations[~filter_points],      over_reconstructed_rotations,   under_reconstructed_rotations])
    
    return GaussianCloud(
        means3D=new_means3D.detach().requires_grad_(),
        means2D=new_means2D.detach().requires_grad_(),
        scales=new_scales.detach().requires_grad_(),
        opacities=new_opacities.detach().requires_grad_(),
        shs=new_shs.detach().requires_grad_(),
        rotations=new_rotations.detach().requires_grad_(),
    )

#TODO: maybe attaching an image directly with the image may be a nice addon. it would be nice to not have this
# dangling index.
def train(
    gaussian_cloud: GaussianCloud,
    train_dataset: list[View],
    test_dataset: list[View],
    *,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] 
        = lambda original, rendered, _lambda=0.2: _lambda * DSSIM(original, rendered) + (1-_lambda) * L1(original, rendered),
    optimizer_factory: Callable[[GaussianCloud], torch.optim.Optimizer] 
        = lambda g_cloud, lr=1e-4: torch.optim.Adam(g_cloud.parameters.values(), lr),
    adaptive_control_frequency: int = 100,
    epochs: int = 300,
) -> tuple[list[float], list[float]]:
    """Main training loop.
    Trains the Gaussian Splatting with adaptive density control.

    Args: 
        gaussian_cloud: Gaussian cloud for the training.
            Will be modified inplace during training and contains the result trained cloud.
        train_dataset: Training dataset as a list of Views to be considered.
        test_dataset: Training dataset as a list of Views to be considered.
        
        loss_function ((original_image, rendered_image) -> tensor) : loss function for
            the gradient descent.
            default: lambda * DSSIM + (1-lambda) * L1
        optimizer_factory: create an optimizer from the gaussian cloud.
            A factory is needed because the cloud and parameters change during ADC.
            default: Adam(cloud_parameters, learning_rate=1e-4)
        adaptative_control_frequency: at every adaptative_control_frequency we perform ADC.
        epochs: total number of epochs on which to train.

    Returns:
        train_loss: list of detaches train losses (floats)
        test_loss: list of detaches test losses (floats)

    """
    training = [
        (create_rasterizer(view), torch.tensor(view.image, dtype=torch.float, device="cuda").permute(2, 0, 1))
        for view in train_dataset
    ]
    testing = [
        (create_rasterizer(view), torch.tensor(view.image, dtype=torch.float, device="cuda").permute(2, 0, 1))
        for view in test_dataset
    ]

    optimizer = optimizer_factory(gaussian_cloud)
    train_loss = []
    test_loss = []
    
    for epoch in range(1, epochs):
        optimizer.zero_grad()
        epoch_train_loss = 0
        
        random.shuffle(training)
        for model, image in training:
            out_img, _ = model(**gaussian_cloud.parameters)
            loss = loss_function(image, out_img)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(training)
        train_loss.append(epoch_train_loss)

        epoch_loss = 0
        for model, image in testing:
            with torch.no_grad():
                out_img, _ = model(**gaussian_cloud.parameters)
                loss = loss_function(image, out_img)
                epoch_loss += loss.item()

        epoch_loss /= len(testing)
        test_loss.append(epoch_loss)

        if epoch % adaptive_control_frequency == 0:
            gaussian_cloud = adaptive_density_control(gaussian_cloud)
            optimizer = optimizer_factory(gaussian_cloud)
        
        if epoch % 10 == 0:
            print(f"{epoch:4}:  loss={epoch_loss:.6f}")
    
    return train_loss, test_loss