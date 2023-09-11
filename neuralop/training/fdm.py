import numpy as np
import torch
import torch.nn.functional as F


class FDM:
    """
    Finite Difference Method (FDM) class for computing differentiation, gradients and Laplacians using central difference upon a grid.

    Paremeters/Atrributes
    ----------
    x : torch.Tensor or None, optional
        Input tensor on which differentiation will be performed. If not provided, it should be passed when calling the methods.
    h : float or list[float] or None, optional
        Step size for differentiation. If a single float is provided, it will be used for all dimensions. If a list is provided, it should match the number of dimensions of `x`. If not provided, it should be passed when calling the methods.
    """

    x = None
    h = None

    def __init__(x: torch.Tensor | None = None, h: float | list[float] | None = None):
        FDM.x = x
        FDM.h = h

    @staticmethod
    def diff(
        x: torch.Tensor, h: float, dim: int, fix_bnd: bool = False
    ) -> torch.Tensor:
        """
        Compute the central difference along a specific dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on which differentiation will be performed.
        h : float
            Step size for differentiation.
        dim : int
            Dimension along which differentiation will be performed.
        fix_bnd : bool, optional
            Whether to apply boundary corrections. If True, boundary corrections are applied. Default is False.

        Returns
        -------
        torch.Tensor
            Tensor representing the central difference along the specified dimension.
        """

        # Check arguments are passed correctly
        x = FDM.x if x is None else x
        h = FDM.h if h is None else h

        assert x is not None, "x cannot be not defined"
        assert h is not None, "h cannot be not defined"

        # Compute the central difference by shifting the grid along the specified dimension
        difference = (torch.roll(x, -1, dims=dim) - torch.roll(x, 1, dims=dim)) / (
            2.0 * h
        )

        # Apply boundary corrections if needed
        if fix_bnd:
            slices_front = [slice(None)] * len(x.shape)
            slices_back = [slice(None)] * len(x.shape)
            slices_front[dim] = 0
            slices_back[dim] = -1

            difference[tuple(slices_front)] = (
                x[tuple(slices_front)] - x[tuple(slices_back)]
            ) / h
            difference[tuple(slices_back)] = (
                x[tuple(slices_back)] - x[tuple(slices_front)]
            ) / h

        return difference

    @staticmethod
    def grad(
        x: torch.Tensor, h: float | list[float], fix_bnd: bool | list[bool] = None
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute the central difference for each dimension of the grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on which differentiation will be performed.
        h : float or list[float]
            Step size(s) for differentiation.
        fix_bnd : bool or list[bool], optional
            List specifying whether to apply boundary corrections for each dimension. If a single bool is provided, it will be used for all dimensions. Default is None, which means no boundary corrections are applied.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tuple of tensors representing the central difference along each dimension.
        """
        num_dims = len(x.shape)

        # Convert h to a list if it's a single float value
        if isinstance(h, float):
            h = [h] * num_dims

        # Initialize fix_bnd to false if it's None
        if fix_bnd is None:
            fix_bnd = [False] * num_dims

        gradients = [FDM.diff(x, h[i], i, fix_bnd[i]) for i in range(num_dims)]
        return tuple(gradients)

    @staticmethod
    def laplacian(
        x: torch.Tensor, h: float | list[float], fix_bnd: bool | list[bool] = None
    ) -> torch.Tensor:
        """
        Compute the Laplacian for the tensor using central difference.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on which the Laplacian will be computed.
        h : float or list[float]
            Step size(s) for differentiation.
        fix_bnd : bool or list[bool], optional
            List specifying whether to apply boundary corrections for each dimension. If a single bool is provided, it will be used for all dimensions. Default is None, which means no boundary corrections are applied.

        Returns
        -------
        torch.Tensor
            Tensor representing the Laplacian of the input tensor.
        """
        num_dims = len(x.shape)

        # Convert h to a list if it's a single float value
        if isinstance(h, float):
            h = [h] * num_dims

        # Initialize fix_bnd to false if it's None
        if fix_bnd is None:
            fix_bnd = [False] * num_dims

        # Compute the second derivative for each dimension and sum them up
        laplace = sum(
            FDM.diff(FDM.diff(x, h[i], i, fix_bnd[i]), h[i], i, fix_bnd[i])
            for i in range(num_dims)
        )
        return laplace

    @staticmethod
    def central_diff_1d(x, h, fix_x_bnd=False) -> torch.Tensor:
        """
        Compute gradient for the 1D grid using central difference
        """
        return FDM.grad(x, h, fix_bnd=[fix_x_bnd])

    @staticmethod
    def central_diff_2d(
        x, h, fix_x_bnd=False, fix_y_bnd=False
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute gradient for the 2D grid using central difference
        """
        return FDM.grad(x, h, fix_bnd=[fix_x_bnd, fix_y_bnd])

    @staticmethod
    def central_diff_3d(
        x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False
    ) -> tuple[torch.Tensor, ...]:
        """
        Compute gradient for the 3D grid using central difference
        """
        return FDM.grad(x, h, fix_bnd=[fix_x_bnd, fix_y_bnd, fix_z_bnd])
