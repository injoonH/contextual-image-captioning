import os
from io import BytesIO
from typing import Union

import pandas as pd
import requests
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMG_SAVE_PATH = "data/images"
REDDIT_JSON_PATH = "data/test_small.json"
REDDIT_URL_PREFIX = "https://i.redd.it/"


def fetch_img(img_src: str) -> Union[Image.Image, None]:
    res = requests.get(img_src)
    if res.status_code == 404:
        return None
    img = Image.open(BytesIO(res.content))
    return img


def save_image_to_tensor(row: pd.Series, transformer: transforms.Compose) -> None:
    img = fetch_img(row["source"])
    if img is None:
        return
    img_tensor = transformer(img)

    file_name = f"{row['id']}.pt"
    path = os.path.join(IMG_SAVE_PATH, file_name)
    torch.save(img_tensor, path)


def save_images(data: pd.DataFrame, transformer: transforms.Compose, path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    tqdm.pandas()
    data.progress_apply(save_image_to_tensor, axis=1, args=(transformer,))


def main() -> None:
    data = pd.read_json(REDDIT_JSON_PATH)
    data["source"] = data["image_hash"].apply(lambda x: REDDIT_URL_PREFIX + x)

    transformer = transforms.Compose([transforms.ToTensor()])
    save_images(data, transformer, IMG_SAVE_PATH)


if __name__ == "__main__":
    main()
