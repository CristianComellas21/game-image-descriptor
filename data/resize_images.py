from skimage import io
from PIL import Image
from pathlib import Path
import numpy as np

def black_background_thumbnail(path_to_image, thumbnail_size=(224,224)):
    background = Image.new('RGB', thumbnail_size, "black")    
    source_image = Image.open(path_to_image).convert("RGB")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, ((thumbnail_size[0] - w) // 2, (thumbnail_size[1] - h) // 2 ))
    return background


def main():

    output_path = Path('images')
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = Path('base_images').glob('*')
    for image_path in image_paths:
        image = black_background_thumbnail(image_path)
        image.save(output_path / image_path.name)

if __name__ == '__main__':
    main()