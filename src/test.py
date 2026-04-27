"""
test.py — Quick test script for image augmentation.
"""

from imageaugmentation import dataGen

if __name__ == '__main__':
    # Test augmentation on a sample image
    dataGen('assets/2021CS002.jpg', save_dir='dataset/', n_images=10)
    print("Augmentation test complete. Check dataset/ folder.")
