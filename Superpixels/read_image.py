from czifile import CziFile
import numpy as np
import matplotlib.pyplot as plt


def readCZI(path):
    path = path

    img = CziFile(path).asarray()
    avg_brightness = float(np.mean(img))
    print("Image:", path)
    print("Average brightness:", avg_brightness)
    arr = np.squeeze(img)   # remove singleton dimensions
    img = arr


    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    return img

    #seg = objectExtractor(image_path=path, image_czi=True, k=k)
