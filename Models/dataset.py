import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, mode: str = 'train', aug: list = []) -> None:
        self.dataset: list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir
        self.mode = mode

        self.categories = {
            'Bags':       [1, 0, 0, 0, 0, 0, 0],
            'Bottomwear': [0, 1, 0, 0, 0, 0, 0],
            'Dress':      [0, 0, 1, 0, 0, 0, 0],
            'Headwear':   [0, 0, 0, 1, 0, 0, 0],
            'Shoes':      [0, 0, 0, 0, 1, 0, 0],
            'Topwear':    [0, 0, 0, 0, 0, 1, 0],
            'Watches':    [0, 0, 0, 0, 0, 0, 1]
        }

        self.__prepare_dataset__()

        post_processing = [
            transforms.CenterCrop((177, 177)),
            transforms.ToTensor()
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((200, 200))] +
            aug +
            post_processing
        )

    def __prepare_dataset__(self) -> None:
        category_files = {}
        category_counts = {}

        # Mengumpulkan file untuk setiap kategori
        for category, label in self.categories.items():
            category_path = os.path.join(self.root_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: Category folder {category} not found!")
                continue

            files = [(os.path.join(category_path, fname), label)
                     for fname in os.listdir(category_path)
                     if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

            category_files[category] = files
            category_counts[category] = len(files)

        # Print distribusi data original
        print("\nOriginal data distribution:")
        for category, count in category_counts.items():
            print(f"{category:.<15} {count:>5} images")

        # Menentukan jumlah minimum sampel untuk setiap split
        min_samples = min(category_counts.values())
        train_size = int(min_samples * 0.6)
        val_size = int(min_samples * 0.3)
        test_size = int(min_samples * 0.1)

        print(f"\nBalanced split sizes:")
        print(f"Train samples per category: {train_size}")
        print(f"Val samples per category: {val_size}")
        print(f"Test samples per category: {test_size}")

        # Membagi dan menyeimbangkan dataset
        for category, files in category_files.items():
            # Mengacak file
            np.random.shuffle(files)

            if self.mode == 'train':
                # Untuk training, gunakan oversampling jika jumlah data kurang
                if len(files) < train_size:
                    files = files * (train_size // len(files) + 1)
                self.dataset.extend(files[:train_size])
            elif self.mode == 'val':
                self.dataset.extend(files[train_size:train_size+val_size])
            else:  # test
                self.dataset.extend(
                    files[train_size+val_size:train_size+val_size+test_size])

        # Print informasi final
        print(f"\nFinal {self.mode} set size: {len(self.dataset)} images")

        if self.mode == 'train':
            # Tambahan augmentasi untuk kelas minoritas
            # kategori dengan sampel sedikit
            minority_categories = ['Dress', 'Headwear']
            additional_samples = []
            for fname, label in self.dataset:
                category = [cat for cat,
                            lbl in self.categories.items() if lbl == label][0]
                if category in minority_categories:
                    # Duplicate samples from minority classes
                    additional_samples.extend([(fname, label)] * 2)

            self.dataset.extend(additional_samples)
            print(
                f"Added {len(additional_samples)} additional samples for minority classes")

        # Mengacak dataset final
        np.random.shuffle(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        try:
            image = Image.open(fpath).convert('RGB')
            image = self.augmentation(image)
            image = (image - image.min()) / (image.max() - image.min())
        except Exception as e:
            print(f"Error loading image {fpath}: {str(e)}")
            # Return alternatif jika gambar error
            image = torch.zeros((3, 177, 177))

        label = torch.Tensor(label)
        return image, label
