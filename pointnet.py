import torch
import torch.nn as nn

class Tnet(nn.Module):
    def __init__(self, dimension, num_points):
        ''' 
        Tnet is a network that learns a transformation matrix for the input point cloud.
        We will optimize it to learn an orthogonal matrix.
        Args:
            dimension: int
                Dimension of the input points.
            num_points: int
                Number of points in the input point cloud.
        '''
        super().__init__()
        self.dimension = dimension
        self.num_points = num_points

        # MLP (dimension, 64, 128, 1024)
        self.conv1 = nn.Conv1d(dimension, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        # MLP (1024, 512, 256, dimension*dimension)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dimension*dimension)

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(num_points)

        # Batch normalization layers for all layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.bn3(torch.relu(self.conv3(x)))

        x = self.max_pool(x).view(batch_size, -1)

        x = self.bn4(torch.relu(self.linear1(x)))
        x = self.bn5(torch.relu(self.linear2(x)))
        x = self.linear3(x)

        x = x.view(-1, self.dimension, self.dimension)
        identity = torch.eye(self.dimension).repeat(batch_size, 1, 1)
        x = x + identity

        return x


class PointNet(nn.Module):
    def __init__(self, num_points, num_global_features):
        '''
        PointNet is a network that classifies point clouds.
        Args:
            num_points: int
                Number of points in the input point cloud.
            num_global_features: int
                Number of global features to be extracted from the point cloud.
        '''
        super().__init__()
        self.num_points = num_points
        self.num_global_features = num_global_features

        # Tnets
        self.tnet1 = Tnet(3, num_points)
        self.tnet2 = Tnet(64, num_points)

        # MLP (64, 64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # MLP (64, 128, num_global_features)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, num_global_features, kernel_size=1)

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(num_points, return_indices=True)

        # Batch normalization layers for all layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(num_global_features)
        

    def forward(self, x):
        batch_size = x.shape[0]
        
        matrix1 = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), matrix1).transpose(2, 1)
        # .transpose(2, 1) is used to make (bs, 3, np) -> (bs, np, 3) for the multiplication

        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))

        matrix2 = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), matrix2).transpose(2, 1)

        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.bn5(torch.relu(self.conv5(x)))

        global_feature, critical_points_indexes = self.max_pool(x)

        global_feature = global_feature.view(batch_size, -1)
        critical_points_indexes = critical_points_indexes.view(batch_size, -1)

        return global_feature, critical_points_indexes


class ClassificationModule(nn.Module):
    def __init__(self, num_points, num_global_features, num_classes):
        '''
        ClassificationModule is a network that classifies the global features extracted from the PointNet.
        Args:
            num_points: int
                Number of points in the input point cloud.
            num_global_features: int
                Number of global features to be extracted from the point cloud.
            num_classes: int
                Number of classes.
        '''
        super().__init__()
        self.num_points = num_points
        self.num_global_features = num_global_features
        self.num_classes = num_classes

        self.pointnet = PointNet(num_points, num_global_features)

        # MLP (num_global_features, 512, 256, num_classes)
        self.linear1 = nn.Linear(num_global_features, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)

        # Batch normalization layers for all layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(num_classes)


    def forward(self, x):
        x, idx = self.pointnet(x)

        x = self.bn1(torch.relu(self.linear1(x)))
        x = self.bn2(torch.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.bn3(torch.relu(self.linear3(x)))

        return x, idx
