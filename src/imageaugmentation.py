import os
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, load_img, save_img
)

def dataGen(image_path: str, save_dir: str = 'dataset/', n_images: int = 15):
    os.makedirs(save_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image = load_img(image_path, target_size=(224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)

    # Ensure unique filenames
    existing = len([f for f in os.listdir(save_dir) if f.startswith("aug_")])

    for i, batch in enumerate(datagen.flow(x, batch_size=1)):
        save_img(os.path.join(save_dir, f'aug_{existing + i + 1}.jpg'), batch[0])
        if i + 1 >= n_images:
            break

    print(f"[augment] {n_images} augmented images saved to {save_dir}")