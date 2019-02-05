import torch
from torchvision import datasets
from torchvision import transforms
from .dataset import Dataset, get_meta_data


def load_me(data_root, train_file, val_file, batch_size=32, num_workers=4):
	train_paths, train_labels = get_meta_data(train_file)
	train_transforms = transforms.Compose([transforms.Resize(234, 240),
	                                       transforms.RandomRotation(degrees=(-8, 8)),
	                                       transforms.RandomCrop(224, 224),
	                                       transforms.ToTensor()])
	train_dataset = Dataset(root=data_root,
	                        img_paths=train_paths,
	                        img_labels=train_labels,
	                        transform=train_transforms)

	val_paths, val_labels = get_meta_data(val_file)
	val_transforms = transforms.Compose([transforms.Resize(234, 240),
	                                     transforms.CenterCrop(224, 224),
	                                     transforms.ToTensor()])
	val_dataset = Dataset(root=data_root,
	                      img_paths=val_paths,
	                      img_labels=val_labels,
	                      transform=val_transforms)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=batch_size,
	                                           num_workers=num_workers,
	                                           shuffle=True)

	val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
	                                         batch_size=batch_size,
	                                         num_workers=num_workers,
	                                         shuffle=True)
	return train_loader, val_loader
