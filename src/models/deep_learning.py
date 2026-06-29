import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Reference: Lawhern et al., 2018 (https://arxiv.org/abs/1611.08024)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_times: int = 1001,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_times = n_times

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.temporal_conv = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Conv: filter size is (n_channels, 1), mapping F1 to F1*D
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv
        # Separable conv = Depthwise + Pointwise.
        # Depthwise: kernel (1, 16) with groups=F1*D
        self.separable_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        # Pointwise: kernel (1, 1) mapping F1*D to F2
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Compute the size of the features before the dense classifier
        # We can pass a dummy tensor to compute this dynamically
        self._num_features = self._get_flattened_size()

        # Classifier
        self.classifier = nn.Linear(self._num_features, n_classes)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.temporal_conv(x)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.dropout1(x)

            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.dropout2(x)

            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_channels, n_times)
        # Add channel dimension: (batch_size, 1, n_channels, n_times)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet: A shallow convolutional network for EEG classification.
    Reference: Schirrmeister et al., 2017 (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_times: int = 1001,
        n_filters_temporal: int = 40,
        filter_time_length: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_times = n_times

        # Temporal Conv
        self.temporal_conv = nn.Conv2d(
            1, n_filters_temporal, kernel_size=(1, filter_time_length), bias=True
        )
        # Spatial Conv: convolves over channels
        self.spatial_conv = nn.Conv2d(
            n_filters_temporal,
            n_filters_temporal,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_filters_temporal)
        # In ShallowConvNet, the activation is squaring: x^2
        # Followed by average pooling and log(x)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout_rate)

        self._num_features = self._get_flattened_size()
        self.classifier = nn.Linear(self._num_features, n_classes)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.bn(x)
            # Squaring activation
            x = x**2
            x = self.pool(x)
            # Log activation
            x = torch.log(torch.clamp(x, min=1e-6))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)

        # Squaring activation
        x = x**2
        x = self.pool(x)
        # Log activation
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepConvNet(nn.Module):
    """
    DeepConvNet: A deep convolutional network for EEG classification.
    Reference: Schirrmeister et al., 2017 (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        n_times: int = 1001,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_times = n_times

        # Block 1
        self.conv1_temporal = nn.Conv2d(1, 25, kernel_size=(1, 10), bias=True)
        self.conv1_spatial = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 2
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 10), bias=True)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 10), bias=True)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        # Block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), bias=True)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.dropout = nn.Dropout(dropout_rate)

        self._num_features = self._get_flattened_size()
        self.classifier = nn.Linear(self._num_features, n_classes)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = F.elu(self.bn1(self.conv1_spatial(self.conv1_temporal(x))))
            x = self.pool1(x)
            x = F.elu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = F.elu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            x = F.elu(self.bn4(self.conv4(x)))
            x = self.pool4(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = F.elu(self.bn1(self.conv1_spatial(self.conv1_temporal(x))))
        x = self.pool1(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
