import logging
import os
from io import BytesIO
from typing import List, Union

import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import ConvNextFeatureExtractor, ResNetModel
from transformers import logging as trans_logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention

SAVE_PATH = "data/sample"
IMG_SAVE_PATH = os.path.join(SAVE_PATH, "images")
FEAT_SAVE_PATH = os.path.join(SAVE_PATH, "features")
REDDIT_JSON_PATH = "data/test_small.json"
REDDIT_URL_PREFIX = "https://i.redd.it/"
PRE_TRAINED_MODEL = "microsoft/resnet-152"


def fetch_img(img_src: str) -> Union[Image.Image, None]:
    res = requests.get(img_src)
    if res.status_code == 404:
        return None
    img = Image.open(BytesIO(res.content))
    return img


def save_image_to_tensor(
    row: pd.Series,
    transformer: transforms.ToTensor,
    img_ids: List[str],
    img_tensors: List[torch.Tensor],
) -> None:
    img_id, img_src = row["id"], row["source"]

    img = fetch_img(img_src)
    if img is None:
        return
    img_tensor = transformer(img)
    if img_tensor.size(0) != 3:  # The number of channels must be three
        return

    file_name = f"{img_id}.pt"
    path = os.path.join(IMG_SAVE_PATH, file_name)
    torch.save(img_tensor, path)

    img_ids.append(img_id)
    img_tensors.append(img_tensor)


def save_image_features(
    img_ids: List[str],
    img_tensors: List[torch.Tensor],
    feature_extractor: ConvNextFeatureExtractor,
    model: ResNetModel,
) -> None:
    logging.info(f"Extracting features of {len(img_ids)} images")
    feat_ext_images = feature_extractor(img_tensors, return_tensors="pt")
    with torch.no_grad():
        outputs: BaseModelOutputWithPoolingAndNoAttention = model(**feat_ext_images)
    features = outputs.pooler_output.view(-1, 2048)
    logging.info("Succeed extracting features from images")

    logging.info("Saving extracted features ...")
    for img_id, feature in tqdm(zip(img_ids, features), total=len(img_ids)):
        file_name = f"{img_id}.pt"
        path = os.path.join(FEAT_SAVE_PATH, file_name)
        torch.save(feature, path)


def save_images(data: pd.DataFrame) -> None:
    # Create directories
    if not os.path.exists(IMG_SAVE_PATH):
        os.makedirs(IMG_SAVE_PATH)
    if not os.path.exists(FEAT_SAVE_PATH):
        os.makedirs(FEAT_SAVE_PATH)

    transformer = transforms.ToTensor()
    img_ids: List[str] = []
    img_tensors: List[torch.Tensor] = []

    tqdm.pandas()
    data.progress_apply(
        save_image_to_tensor, axis=1, args=(transformer, img_ids, img_tensors)
    )

    feature_extractor = ConvNextFeatureExtractor.from_pretrained(PRE_TRAINED_MODEL)
    model = ResNetModel.from_pretrained(PRE_TRAINED_MODEL)

    save_image_features(img_ids, img_tensors, feature_extractor, model)


def main() -> None:
    data = pd.read_json(REDDIT_JSON_PATH)
    data["source"] = data["image_hash"].apply(lambda x: REDDIT_URL_PREFIX + x)

    # Hide warnings
    trans_logging.set_verbosity_error()

    logging.info(f"Processing {len(data)} reddit data ...")
    save_images(data)
    saved_names = os.listdir(IMG_SAVE_PATH)
    saved_ids = [os.path.splitext(name)[0] for name in saved_names]
    saved_data = data[data["id"].isin(saved_ids)]
    saved_data.to_csv(os.path.join(SAVE_PATH, "summary.csv"), index=False)
    logging.info(f"{len(saved_data)} image data were saved successfully")


if __name__ == "__main__":
    logging.basicConfig(
        filename="image-scraper.log",
        format="%(asctime)s %(message)s",
        level=logging.INFO,
    )
    main()
