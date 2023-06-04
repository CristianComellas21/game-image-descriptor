from __future__ import annotations

from pathlib import Path
from typing import List
import json
import random
from tqdm import tqdm
import numpy as np


BASE_PATH = Path('annotations')
MODES = ['train', 'val', 'test']


def create_folder(name: str) -> Path:
    """Create folder if it doesn't exist and return its Path object"""
    path = BASE_PATH / Path(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():

    data = {
        'train': [],
        'val': [],
        'test': []

    }

    # Set partitions size (train, val, test)
    train_partition: float = 0.85
    val_partition: float = 0.1
    test_partition: float = 1 - (train_partition + val_partition)

    # Load image paths
    images = list(Path('images').glob('*'))
    images = sorted(images, key=lambda x: int(x.stem.split('_')[0]))

    # Define partitions
    modes = np.random.default_rng(seed=21).choice(MODES, size=len(images), p=[train_partition, val_partition, test_partition])

    # Load descriptions and build annotations
    n_elements = len(images)
    with open('descriptions.txt', 'r') as f:
        descriptions: List[str] = f.read().split('\n-\n')[:-1]

        # Build elements
        for idx, (image, description) in (progress := tqdm(enumerate(zip(images, descriptions)))):

            progress.set_description(f"Processing image {idx+1:05}/{n_elements}...")

            # Build annotations
            data[modes[idx]].append({
                'image': str(image),
                'image_id': f"{idx:05}",
                'caption': description
                }
                )


    # Save annotations
    annotations_path = Path('annotations')

    print(f"Saving annotations to {annotations_path}...")

    json.dump(data['train'], open(annotations_path / Path('train.json'), 'w'), ensure_ascii=False)
    json.dump(data['val'], open(annotations_path / Path('val.json'), 'w'), ensure_ascii=False)
    json.dump(data['test'], open(annotations_path / Path('test.json'), 'w'), ensure_ascii=False)











if __name__ == '__main__':
    main()