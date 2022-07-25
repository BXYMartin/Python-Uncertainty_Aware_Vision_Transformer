import matplotlib.pyplot as plt
import glob
import os
import numpy as np
# level80_case0035_image


base_path = os.path.join("output_original_single_channel", "predictions-patch-white")
storage_path = os.path.join("output_original_single_channel", "predictions-patch-white-processed")
if not os.path.exists(storage_path):
    os.mkdir(storage_path)
for level in range(48, 90):
    images = []
    for filename in glob.glob(os.path.join(base_path, f"level{level}_case0035_*_prediction.npy")):
        # im = plt.imread(filename)
        im = np.load(filename)
        im[im != 0] = 1
        images.append(im)
    preds = np.stack(images)
    preds.shape
    variance = np.var(preds, axis=0)
    # variance = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))
    print(f"{level}")
    plt.imshow(variance, cmap="magma")
    plt.imsave(os.path.join(storage_path, os.path.splitext(os.path.basename(filename))[0].replace("prediction", "uncertainty") + ".jpg"), variance, cmap="magma")
    plt.show()
