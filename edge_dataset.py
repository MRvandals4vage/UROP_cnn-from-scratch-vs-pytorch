import random

# Synthetic dataset generator for edge-friendly workloads using pure Python and lists
# No external dependencies.

def generate_image(height, width, channels=1, pattern="gradient", noise_level=0):
    """
    Generate a single synthetic image as a list[channels][height][width].
    Supported patterns: 'gradient', 'checker', 'dot', 'bars'
    """
    img = []
    for c in range(channels):
        channel = []
        for i in range(height):
            row = []
            for j in range(width):
                if pattern == "gradient":
                    v = (i + j) / float(height + width)
                elif pattern == "checker":
                    v = 1.0 if ((i // 2 + j // 2) % 2 == 0) else 0.0
                elif pattern == "dot":
                    v = 1.0 if (i == height // 2 and j == width // 2) else 0.0
                elif pattern == "bars":
                    v = 1.0 if (j % 3 == 0) else 0.0
                else:
                    v = 0.0
                if noise_level > 0:
                    v += (random.random() - 0.5) * 2 * noise_level
                row.append(v)
            channel.append(row)
        img.append(channel)
    return img


def generate_synthetic_dataset(num_samples=10, height=8, width=8, channels=1, classes=2):
    """
    Generate a small dataset: returns (images, labels)
    images: list of images; each image is list[channels][height][width]
    labels: list of ints in [0, classes-1]
    Patterns are assigned per class for simplistic classification.
    """
    pattern_by_class = ["gradient", "checker", "dot", "bars"]
    images = []
    labels = []
    for n in range(num_samples):
        label = n % classes
        pattern = pattern_by_class[label % len(pattern_by_class)]
        noise = 0.05 if (n % 3 == 0) else 0.0
        img = generate_image(height, width, channels, pattern=pattern, noise_level=noise)
        images.append(img)
        labels.append(label)
    return images, labels


def pretty_print_image(img):
    """Print first channel of image for quick inspection."""
    ch0 = img[0]
    for row in ch0:
        print([f"{v:0.2f}" for v in row])


def main():
    print("--- Edge Synthetic Dataset Demo ---")
    images, labels = generate_synthetic_dataset(num_samples=4, height=6, width=6, channels=1, classes=2)
    for idx, (img, lab) in enumerate(zip(images, labels)):
        print(f"\nSample {idx} Label={lab}")
        pretty_print_image(img)

if __name__ == "__main__":
    main()
