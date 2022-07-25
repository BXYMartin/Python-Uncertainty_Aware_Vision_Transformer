import os, sys

split = "val"

with open(os.path.join("lists", "lists_LIDC", f"{split}.txt"), "w") as list_file:
    for sample_dir in os.listdir(os.path.join("data", "LIDC", split, "images")):
        if sample_dir.startswith("LIDC-IDRI"):
            for sample_file in os.listdir(os.path.join("data", "LIDC", split, "images", sample_dir)):
                list_file.write(f"{sample_dir}/{os.path.splitext(sample_file)[0]}\n")
                

