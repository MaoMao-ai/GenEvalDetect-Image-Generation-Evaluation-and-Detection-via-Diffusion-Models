import os
from compute_msfid import compute_fid_and_msfid

def main():
    real_dir = r"D:\VS Projects\ECE-580\Proj\MSFID\real"
    fake_dir = r"D:\VS Projects\ECE-580\Proj\MSFID\fake"

    print("✅ Starting MSFID Evaluation...")
    fid, msfid = compute_fid_and_msfid(real_dir, fake_dir, model_path=None, max_images=5)
    print("✅ Evaluation Complete:")
    print(f"    FID   : {fid:.4f}")
    print(f"    MSFID: {msfid:.4f}")

if __name__ == "__main__":
    main()
