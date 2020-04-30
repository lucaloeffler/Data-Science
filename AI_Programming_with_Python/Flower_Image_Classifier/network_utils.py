from torchvision import models
from torch import nn
import torch


def build_network(arch, hidden_dim, output_dim, drop_prob):
    if arch == 'resnet18':
        print("Using pretrained resnet18")
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        print("Using pretrained resnet34")
        model = models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        print("Using pretrained resnet50")
        model = models.resnet50(pretrained=True)
    elif arch == 'resnet101':
        print("Using pretrained resnet101")
        model = models.resnet101(pretrained=True)
    else:
        print(f"Im sorry but {arch} is not a valid model. Using ResNet18 by default.")
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, output_dim),
        nn.LogSoftmax(dim=1)
    )

    model.fc = classifier
    return model


def get_loss_function():
    return nn.NLLLoss()


def get_optimizer(model, lr):
    return torch.optim.Adam(model.fc.parameters(), lr)


def save_model(model, save_dir, arch, epochs, lr, hidden_units):
    if save_dir is '':
        save_path = f'./checkpoint-{arch}.pth'
    else:
        save_path = save_dir + f'/checkpoint-{arch}.pth'

    model.cpu()

    checkpoint = {
        'arch': 'resnet18',
        'hidden_dim': hidden_units,
        'epochs': epochs,
        'lr': lr,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, save_path)


def load_model(checkpoint_path):
    trained_model = torch.load(checkpoint_path)
    model = build_network(arch=trained_model['arch'], hidden_dim=trained_model['hidden_dim'],
                          output_dim=102, drop_prob=0)

    model.class_to_idx = trained_model['class_to_idx']
    model.load_state_dict(trained_model['state_dict'])
    print(f"Successfully loaded model with arch {trained_model['arch']}")
    return model
