import os
import tarfile
import random

# === 配置：你本地的图像根目录（解压路径 & 输出路径） ===
DATA_ROOT = "D:/VS Projects/ECE-580/Proj/images"
OUT_ROOT = "D:/VS Projects/ECE-580/Proj/datasets"
SPLITS = ["train", "test", "val"]
MODES = ["adm"]  # 如有 sdv1，可加进去

def unpack_tar(tar_path, output_dir, fraction=0.2):
    """仅解压部分文件（默认20%）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        n_total = len(members)
        n_extract = max(1, int(n_total * fraction))
        sampled = random.sample(members, n_extract)
        tar.extractall(path=output_dir, members=sampled)
    print(f"✅ Extracted {n_extract}/{n_total} from {os.path.basename(tar_path)}")

# === 主执行逻辑：解压===
for split in SPLITS:
    for mode in MODES:
        print(f"\n🔍 Evaluating split={split}, mode={mode}...")

        real_tar = os.path.join(DATA_ROOT, split, "imagenet", "real.tar.gz")
        fake_tar = os.path.join(DATA_ROOT, split, "imagenet", f"{mode}.tar.gz")

        real_dir = os.path.join(OUT_ROOT, "imagenet", split, "real")
        fake_dir = os.path.join(OUT_ROOT, "imagenet", split, mode)

        # 解压 20%
        unpack_tar(real_tar, real_dir, fraction=0.01)
        unpack_tar(fake_tar, fake_dir, fraction=0.01)