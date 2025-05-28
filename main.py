import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import copy
import random
import torch.nn.functional as F
from download import download_dataset


class ImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None, num_samples=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "ImageNet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        samples_dir = os.path.join(root, split)
        image_to_label = {}
        with open(os.path.join(root, "ImageNet_val_label.txt"), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_to_label[parts[0]] = parts[1]
        self.val_to_syn = image_to_label
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
        if num_samples is not None and num_samples < len(self.samples):
            indices = random.sample(range(len(self.samples)), num_samples)
            self.samples = [self.samples[i] for i in indices]
            self.targets = [self.targets[i] for i in indices]
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]
    

def accuracy(model): # top@1 accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            y_pred = model(x)
            y_pred = F.softmax(y_pred, dim=-1)
            correct += (y_pred.argmax(axis=1) == y).sum().item()
            total += len(y)
    print("Accuracy top@1: ", correct / total)


if __name__ == "__main__":
    random.seed(0)
    download_dataset()
    images_num = None # to select random subset of images to run test on

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    dataset = ImageNetDataset('imagenet', "val", val_transform, num_samples=images_num)
    dataloader = DataLoader(
                dataset,
                batch_size=16,
                num_workers=2,
                shuffle=False,
                drop_last=False,
                pin_memory=False
            )

    model = torchvision.models.resnet50(weights="DEFAULT")
    model.eval()
    accuracy(model)
