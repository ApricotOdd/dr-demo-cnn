from torchvision import transforms as T


def get_transforms(image_size=224, train=True):
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])