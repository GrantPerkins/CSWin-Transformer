import matplotlib.pyplot as plt
from pathlib import Path
import cv2

cam_path = Path(r"C:\Users\gcper\Code\Thesis\CSWin-Transformer\cam")
src_path = Path(r"C:\Users\gcper\Downloads\full_size_pu(1)\full_size_pu\val")

for path in (cam_path/"gradcam").iterdir():
    stage = path.stem[6]
    source_img_path = src_path / stage / f"{path.stem}.jpg"
    fig, ax = plt.subplots(2, 1, figsize=(5,10))

    ax[0].axis("off")
    ax[0].set_title(f"PI Stage {stage}")
    img = cv2.imread(str(source_img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0].imshow(img)

    ax[1].set_title(f"GradCAM")
    ax[1].axis("off")
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[1].imshow(img)
    plt.savefig(str(cam_path/"combined"/path.name))
    plt.close()

