import os
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn

def segmenatation(image_path, output_dir_path):
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )


    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

    # Number of classes found in image
    classes = np.unique(pred_seg)

    # Extracting filename from image
    file_name = os.path.basename(image_path)
    # Construct output file path
    output_file_path = os.path.join(output_dir_path, f"{file_name}")

    # Iterate over each class and crop the corresponding areas from the original image
    for class_id in classes:
        mask = (pred_seg == class_id).astype(np.uint8)

        # Apply the mask to the original image
        masked_image = np.array(image) * mask[:, :, np.newaxis]

        # Convert the masked image back to PIL format
        cropped_area = Image.fromarray(masked_image)

        if class_id == 2:
            cropped_area.save(os.path.join(output_dir_path, f"{file_name}"))

    return output_file_path