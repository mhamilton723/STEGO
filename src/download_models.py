from pathlib import Path

import wget

models_dir = Path("..") / "models"
models_dir.mkdir(exist_ok=True)
model_url_root = (
    "https://marhamilresearch4.blob.core.windows.net/stego-public/models/models/"
)
model_names = [
    "moco_v2_800ep_pretrain.pth.tar",
    "model_epoch_0720_iter_085000.pth",
    "picie.pkl",
]

saved_models_dir = Path("..") / "saved_models"
saved_models_dir.mkdir(exist_ok=True)
saved_model_url_root = (
    "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
)
saved_model_names = [
    "cityscapes_vit_base_1.ckpt",
    "cocostuff27_vit_base_5.ckpt",
    "picie_and_probes.pth",
    "potsdam_test.ckpt",
]

target_files = [models_dir / mn for mn in model_names] + [
    saved_models_dir / mn for mn in saved_model_names
]

target_urls = [model_url_root + mn for mn in model_names] + [
    saved_model_url_root + mn for mn in saved_model_names
]

for target_file, target_url in zip(target_files, target_urls):
    if not Path(target_file).exists():
        print(f"\nDownloading file from {target_url}")
        wget.download(target_url, target_file)
    else:
        print(f"\nFound {target_file}, skipping download")
