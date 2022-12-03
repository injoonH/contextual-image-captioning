# Contextual Image Captioning

![Gitmoji][1]
[![Code style: black][2]](https://github.com/psf/black)
[![Imports: isort][3]](https://pycqa.github.io/isort/)

Refactored codes for the paper "Exploiting Image-Text Synergy for Contextual
Image Captioning," published at EACL Workshop LANTERN 2021.

## Prerequisites

- Python 3.8
- CUDA 11.2

## Usage

1. Install packages.

   Follow [PyTorch Instructions][4] to install `torch` & `torchvision`.

   ```bash
   python -m pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Run `src/image_scraper.py` to download image data from Reddit. Update
   constants defined at the top of the file before running if needed.

   ```bash
   python src/image_scraper.py
   ```

## Contributors

- @injoonH

## References

1. Sreyasi Nag Chowdhury, Rajarshi Bhowmik, Hareesh Ravi, Gerard de Melo, Simon
   Razniewski, and Gerhard Weikum. 2021. [Exploiting Image–Text Synergy for
   Contextual Image Captioning][5]. In _Proceedings of the Third Workshop on
   Beyond Vision and LANguage: inTEgrating Real-world kNowledge (LANTERN)_,
   pages 30–37, Kyiv, Ukraine. Association for Computational Linguistics.

[1]: https://img.shields.io/badge/gitmoji-%20😜%20😍-ffdd67?style=flat-square "Shield-Gitmoji"
[2]: https://img.shields.io/badge/code%20style-black-000000?style=flat-square "Shield-Black"
[3]: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat-square&labelColor=ef8336 "Shield-isort"
[4]: https://pytorch.org/get-started/locally/ "PyTorch Installation"
[5]: https://aclanthology.org/2021.lantern-1.3 "Paper"
