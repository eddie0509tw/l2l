import learn2learn as l2l
from torchvision import transforms
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)
from collections import namedtuple

BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))

def create_miniimgnat(train_ways, train_samples, test_ways, test_samples, num_tasks=-1):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),          # Convert the PIL Image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for ImageNet
                            std=[0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    path = './datasets'
    # Create Tasksets using the benchmark interface
    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=path,
        mode='train',
        download=True,
        transform=train_transform,
    )
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=path,
        mode='validation',
        download=True,
        transform=valid_transform,
    )
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=path,
        mode='test',
        download=True,
        transform=test_transform,
    )

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        NWays(train_dataset, train_ways),
        KShots(train_dataset, train_samples),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_transforms = [
        NWays(valid_dataset, test_ways),
        KShots(valid_dataset, test_samples),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    test_transforms = [
        NWays(test_dataset, test_ways),
        KShots(test_dataset, test_samples),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]

    train_tasks = l2l.data.Taskset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.Taskset(
        dataset=valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.Taskset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_tasks,
    )
    tasksets = BenchmarkTasksets(train_tasks, validation_tasks, test_tasks)
    return tasksets