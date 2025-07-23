import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def detect_chessboard_corners(
    image_path: str,
    board_size=(8, 11),
    display_scale_variants=(1.0, 0.5, 1.5),
    clahe_clip=2.0,
    clahe_grid=(8, 8),
):
    """
    Detect chessboard corners in an image using multiple scales and colour channels.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    board_size : tuple, optional
        Expected number of internal corners per chessboard row and column (cols, rows).
    display_scale_variants : tuple, optional
        Down/up‑scaling factors to try for robustness.
    clahe_clip : float, optional
        CLAHE clip limit (contrast limiting).
    clahe_grid : tuple, optional
        CLAHE tile grid size.

    Returns
    -------
    (np.ndarray | None, np.ndarray | None)
        Tuple of (N×2 float32 corner coordinates in original image space,
        BGR image with drawn corners).  Returns (None, None) if detection fails.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    # Pre‑store original grayscale & CLAHE for later sub‑pixel refinement
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_eq_orig = clahe.apply(gray_orig)

    pattern_sizes = [board_size, board_size[::-1]]  # try both orientations
    found = False
    best_corners = None
    best_pattern = None

    # Iterate over orientation, scale, and individual channels
    for psize in pattern_sizes:
        if found:
            break
        for scale in display_scale_variants:
            if found:
                break
            # Resize for scale variant
            scaled = (
                cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if scale != 1.0
                else img_bgr
            )
            # Prepare grayscale/colour channels to test
            channels = []

            # Equalised grayscale
            gray_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            channels.append(clahe.apply(gray_scaled))

            # Equalised individual B, G, R channels
            for ch in range(3):
                channels.append(clahe.apply(scaled[:, :, ch]))

            # Equalised L* (lightness) from LAB colour space
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            channels.append(clahe.apply(lab[:, :, 0]))

            # Test each prepared channel
            for channel_img in channels:
                ret, corners = cv2.findChessboardCornersSB(
                    channel_img,
                    psize,
                    flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
                )
                if ret:
                    # Rescale corner coords back to original image space
                    if scale != 1.0:
                        corners /= scale
                    best_corners = corners.copy()
                    best_pattern = psize
                    found = True
                    break

    if not found:
        return None, None

    # Sub‑pixel refinement on full‑resolution equalised grayscale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(
        gray_eq_orig,
        best_corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    # Draw detected corners for visual confirmation
    drawn = img_bgr.copy()
    cv2.drawChessboardCorners(drawn, best_pattern, best_corners, True)

    return best_corners.reshape(-1, 2), drawn


# Paths to uploaded images
color_path = "./data/Color.bmp"
ir_path = "./data/IR.bmp"

color_corners, color_drawn = detect_chessboard_corners(color_path, board_size=(8, 11))
ir_corners, ir_drawn = detect_chessboard_corners(ir_path, board_size=(8, 11))

# Save annotated images for download if detection succeeded
out_files = []
if color_drawn is not None:
    out_color_path = "./data/res/Color_annotated.jpg"
    cv2.imwrite(out_color_path, color_drawn)
    out_files.append(out_color_path)

if ir_drawn is not None:
    out_ir_path = "./data/res/IR_annotated.jpg"
    cv2.imwrite(out_ir_path, ir_drawn)
    out_files.append(out_ir_path)

# Display results side‑by‑side
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].set_title("Color image")
axes[0].imshow(
    cv2.cvtColor(
        color_drawn if color_drawn is not None else cv2.imread(color_path),
        cv2.COLOR_BGR2RGB,
    )
)
axes[0].axis("off")

axes[1].set_title("IR image")
axes[1].imshow(
    cv2.cvtColor(
        ir_drawn if ir_drawn is not None else cv2.imread(ir_path),
        cv2.COLOR_BGR2RGB,
    )
)
axes[1].axis("off")
plt.show()

# Report detection status & corner coordinates (first few rows)
def summarise(name, corners):
    if corners is None:
        print(f"{name}: detection FAILED")
    else:
        print(f"{name}: detection SUCCESS ‑‑ {corners.shape[0]} corners found")
        print(corners[: min(10, corners.shape[0])])  # print first 10 rows


summarise("Color", color_corners)
summarise("IR", ir_corners)

# Provide download links for annotated images (if any)
for f in out_files:
    print(f"[Download annotated {Path(f).name}]({f})")
