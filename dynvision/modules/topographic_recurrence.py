import time
import torch
import torch.nn as nn

__all__ = ["LocalLateralConnection", "LocalSeparableConnection"]


def is_square_number(n: int) -> bool:
    if n < 0:
        return False
    sqrt_n = int(torch.sqrt(torch.tensor(n, dtype=torch.int64)))
    is_square = bool(sqrt_n * sqrt_n == n)
    return is_square


def next_square_number(c: int) -> int:
    i = int(torch.sqrt(torch.tensor(c, dtype=torch.int64))) + 1
    return int(i**2)


def extend_to_square_channel_number(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    n_channels = x.size(dim)

    if is_square_number(n_channels):
        return x

    add_n = next_square_number(n_channels) - n_channels
    repeated_slices = x.narrow(dim, n_channels - add_n, add_n).flip(
        dims=[dim]
    )  # reflecting last slices

    return torch.cat((x, repeated_slices), dim=dim)


def get_coordinates(shape: tuple) -> torch.Tensor:
    """
    generates a tensor of given shape with each element containing its coordinates (along an additional dimension).
    """
    *n_channels, dim_y, dim_x = shape
    x = torch.arange(dim_x, dtype=torch.int64)
    y = torch.arange(dim_y, dtype=torch.int64)

    if len(n_channels) > 1:
        raise ValueError("Invalid shape!")

    elif len(n_channels) == 1:
        z = torch.arange(n_channels[0], dtype=torch.int64)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coordinates = torch.stack((zz, yy, xx), dim=-1)

    else:  # no channel dimension
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1)

    return coordinates


def _layer_to_plane(layer: torch.Tensor) -> torch.Tensor:
    """
    Maps a 3d (+coord dim) layer tensor to a 2d plane tensor by arranging the feature maps in a grid.
    """
    n_channels, dim_y, dim_x, dim_coords = layer.shape
    feature_dim = int(torch.sqrt(torch.tensor(n_channels, dtype=torch.float32)))
    plane_dim_x = feature_dim * dim_x
    plane_dim_y = feature_dim * dim_y
    plane = torch.zeros(
        (plane_dim_y, plane_dim_x, dim_coords),
        dtype=torch.int64,
        device=layer.device,
    )

    flip_indices = torch.arange(feature_dim - 1, -1, -1, device=layer.device)

    for y in range(dim_y):
        for x in range(dim_x):
            feature = layer[:, y, x].reshape(feature_dim, feature_dim, dim_coords)

            if x % 2 == 1:
                feature = feature[:, flip_indices]
            if y % 2 == 1:
                feature = feature[flip_indices, :]

            y0, y1 = y * feature_dim, (y + 1) * feature_dim
            x0, x1 = x * feature_dim, (x + 1) * feature_dim
            plane[y0:y1, x0:x1] = feature

    return plane


def _plane_to_layer(
    plane: torch.Tensor, n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Maps a 2D (+coord dim) plane tensor to a 3D layer tensor by extracting feature maps from a grid.
    """
    plane = plane.squeeze()
    plane_dim_y, plane_dim_x, dim_coords = plane.shape
    feature_dim = int(torch.sqrt(torch.tensor(n_channels, dtype=torch.float32)))
    layer = torch.zeros(
        (n_channels, dim_y, dim_x, dim_coords),
        dtype=torch.int64,
        device=plane.device,
    )

    flip_indices = torch.arange(feature_dim - 1, -1, -1, device=plane.device)

    for y in range(dim_y):
        for x in range(dim_x):
            y0, y1 = y * feature_dim, (y + 1) * feature_dim
            x0, x1 = x * feature_dim, (x + 1) * feature_dim
            feature = plane[y0:y1, x0:x1]

            if x % 2 == 1:
                feature = feature[:, flip_indices]
            if y % 2 == 1:
                feature = feature[flip_indices, :]

            layer[:, y, x] = feature.reshape(-1, dim_coords)

    return layer


def create_layer_to_plane_mapping(
    n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Create a lookup table that maps a 3D layer tensor to a 2D plane tensor.
    """
    feature_dim = int(torch.sqrt(torch.tensor(n_channels, dtype=torch.float32)))
    plane_dim_y = feature_dim * dim_y
    plane_dim_x = feature_dim * dim_x

    plane_coordinates = get_coordinates((plane_dim_y, plane_dim_x))
    return _plane_to_layer(plane_coordinates, n_channels, dim_y, dim_x)


def create_plane_to_layer_mapping(
    n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Create a lookup table that maps a 2D plane tensor to a 3D layer tensor.
    """
    layer_coordinates = get_coordinates((n_channels, dim_y, dim_x))

    return _layer_to_plane(layer_coordinates)


def layer_to_plane(layer: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """
    Map a 3D layer tensor to a 2D plane tensor using a lookup table.
    """
    batch_size, n_channels, dim_y, dim_x = layer.shape
    feature_dim = int(torch.sqrt(torch.tensor(n_channels, dtype=torch.float32)))
    plane_dim_x = feature_dim * dim_x
    plane_dim_y = feature_dim * dim_y

    batch_index = (
        torch.arange(batch_size, device=layer.device)
        .view(-1, 1, 1)
        .expand(-1, plane_dim_y, plane_dim_x)
    )
    z_index, y_index, x_index = mapping[:, :, 0], mapping[:, :, 1], mapping[:, :, 2]

    plane = layer[
        batch_index,
        z_index.expand(batch_size, -1, -1),
        y_index.expand(batch_size, -1, -1),
        x_index.expand(batch_size, -1, -1),
    ]
    plane = plane.unsqueeze(1)  # add channel dimension
    return plane


def plane_to_layer(plane: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """
    Map a 2D plane tensor to a 3D layer tensor using a lookup table.
    """
    batch_size, *_, plane_dim_y, plane_dim_x = plane.shape
    n_channels, dim_y, dim_x, _ = mapping.shape

    batch_index = (
        torch.arange(batch_size, device=plane.device)
        .view(-1, 1, 1, 1)
        .expand(-1, n_channels, dim_y, dim_x)
    )
    y_index, x_index = mapping[:, :, :, 0], mapping[:, :, :, 1]

    layer = plane[
        batch_index,
        0,
        y_index.expand(batch_size, -1, -1, -1),
        x_index.expand(batch_size, -1, -1, -1),
    ]
    return layer


class LocalLateralConnection(nn.Module):
    def __init__(
        self, in_channels, dim_y, dim_x, kernel_size, bias=True, device=None, max_weight_init=0.05, **kwargs
    ):
        super(LocalLateralConnection, self).__init__()
        
        self.max_weight_init = max_weight_init

        self.layer_to_plane_mapping = create_layer_to_plane_mapping(
            in_channels, dim_y, dim_x
        )
        self.plane_to_layer_mapping = create_plane_to_layer_mapping(
            in_channels, dim_y, dim_x
        )

        if device is not None:
            self.device = device
            self.layer_to_plane_mapping = self.layer_to_plane_mapping.to(device)
            self.plane_to_layer_mapping = self.plane_to_layer_mapping.to(device)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # determine kernel size for the plane mapping
        feature_dim = int(torch.sqrt(torch.tensor(in_channels, dtype=torch.float32)))
        plane_kernel_size = feature_dim // 4
        if plane_kernel_size % 2 == 0:
            plane_kernel_size += 1
        plane_kernel_size = max(3, plane_kernel_size)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=plane_kernel_size,
            stride=1,
            padding=plane_kernel_size // 2,
            padding_mode="reflect",
            bias=bias,
            device=self.device,
        )

    def _init_parameters(self):
        nn.init.uniform_(self.conv.weight, a=-self.max_weight_init, b=self.max_weight_init)
        

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, dim_y, dim_x = x0.shape
        # x0 = extend_to_square_channel_number(x0)

        p0 = layer_to_plane(x0, self.plane_to_layer_mapping)

        p1 = self.conv(p0)

        x1 = plane_to_layer(p1, self.layer_to_plane_mapping)

        x1 = x1[:, :n_channels]

        return x1

    def trace_forward(self, x0: torch.Tensor) -> torch.jit.ScriptModule:
        return torch.jit.trace(self.forward, x0)


class LocalSeparableConnection(nn.Module):
    def __init__(
        self,
        in_channels,
        dim_y,
        dim_x,
        kernel_size,
        bias=True,
        max_weight_init=0.05,
        **kwargs,
    ):
        super(LocalSeparableConnection, self).__init__()

        self.local_conv = LocalLateralConnection(
            in_channels=in_channels,
            dim_y=dim_y,
            dim_x=dim_x,
            kernel_size=kernel_size,
            bias=bias,
        )

        self.patchy_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=bias,
        )

        max_w_init = max_weight_init
        nn.init.uniform_(self.patchy_conv.weight, a=-max_w_init, b=max_w_init)
        nn.init.uniform_(self.local_conv.conv.weight, a=-max_w_init, b=max_w_init)

    def forward(self, x):
        # local connection are shorter and faster
        h = self.local_conv(x)
        # input from patchy connections comes later
        h = self.patchy_conv(h)
        return h
