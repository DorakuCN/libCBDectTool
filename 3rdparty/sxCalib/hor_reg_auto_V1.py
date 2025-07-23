"""
hor_reg_auto_V1.py
====================

This module re‑implements the `hor_reg_auto_V1` MATLAB calibration routine in
Python.  The purpose of the original code is to estimate a set of
polynomial warp parameters that align an infrared (IR) image with a colour
image.  It performs the following high‑level steps:

1. Reads a plain‑text configuration file to obtain paths to the raw image
   files, camera intrinsics, baseline distances and miscellaneous flags.
2. Loads the IR, colour and depth images from their raw formats.  The IR
   image is read as 16‑bit values, normalised to 8‑bit and optionally
   gamma‑corrected.  A fixed pixel shift of (4,4) pixels is applied to
   account for sensor misalignment depending on the hardware version.
3. The colour image is read from a packed RGB byte stream and converted
   into a greyscale image for corner detection.
4. A chessboard calibration target is detected in both the IR and colour
   images.  The corner detection is delegated to the `detect_chessboard_corners`
   function provided in ``sxCheckBoardDetect.py`` to ensure identical
   behaviour across platforms.
5. Using the detected corner coordinates, a least‑squares fit computes
   quadratic polynomial coefficients mapping IR pixel positions into the
   colour image.  A depth‑dependent term is subtracted from the horizontal
   direction to compensate for baseline disparity between the IR camera,
   colour camera and projector.
6. The resulting coefficients are scaled by ``2**20`` and written as
   integer values to the output parameter file.  Different output
   formatting is selected depending on the hardware version.

This implementation follows the mathematical formulas and data flow of
``hor_reg_auto_V1.m`` as closely as possible.  It intentionally mirrors
variable names from the original code to aid comparison and debugging.

This module is designed for production use - it contains NO visualization
or display code.  All functions are pure computation suitable for
headless environments and automated processing pipelines.

Example
-------

Assuming a configuration file ``hor_reg.txt`` similar to the one used in
the MATLAB code, the following call will perform the registration and
write the integer calibration parameters to the output file listed in the
configuration::

    from hor_reg_auto_V1 import hor_reg_auto
    hor_reg_auto("hor_reg.txt")

The function raises ``RuntimeError`` when an image cannot be loaded or
when the chessboard cannot be detected in either the IR or colour image.

"""

import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import cv2  # type: ignore
import numpy as np

try:
    # Import the chessboard detection helper from the provided module.  If
    # the module cannot be found the ImportError will surface at runtime.
    from sxCheckBoardDetect import detect_chessboard_corners
except Exception as exc:  # pragma: no cover - import errors handled at runtime
    raise ImportError(
        "Failed to import detect_chessboard_corners from sxCheckBoardDetect. "
        "Ensure that sxCheckBoardDetect.py is present in the same directory."
    ) from exc


def _read_config(path: str) -> Tuple[int, int, str, str, str, int, int,
                                     float, float, float, float, str]:
    """Parse the registration configuration file.

    The configuration file is expected to contain one parameter per line in
    the order defined by the original MATLAB implementation.  Blank lines
    are ignored.  Numerical entries are converted to ``int`` or ``float``
    as appropriate.  The focal length of the RGB camera (fx_rgb) is scaled
    by 0.5 to mirror the MATLAB code.

    Parameters
    ----------
    path : str
        Path to the configuration text file.

    Returns
    -------
    tuple
        A tuple containing the parsed configuration parameters in the
        following order::

            (ifshow, version, ir_path, color_path, depth_path,
             ir_width, ir_height, fx_rgb, color_ir_base,
             fx_ir, ir_projector_base, params_path)

    Raises
    ------
    RuntimeError
        If the file cannot be read or the format does not match the
        expected number of lines.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"Configuration file '{path}' does not exist.")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # collect non‑empty lines and strip whitespace
        lines: List[str] = [line.strip() for line in f if line.strip()]
    if len(lines) < 12:
        raise RuntimeError(
            f"Configuration file '{path}' must contain at least 12 non‑empty lines, "
            f"but only {len(lines)} were found."
        )
    # Extract parameters
    ifshow = int(float(lines[0]))
    version = int(float(lines[1]))
    ir_path = lines[2]
    color_path = lines[3]
    depth_path = lines[4]
    ir_width = int(float(lines[5]))
    ir_height = int(float(lines[6]))
    # The MATLAB code multiplies fx_rgb by 0.5 after reading
    fx_rgb = float(lines[7]) * 0.5
    color_ir_base = float(lines[8])
    fx_ir = float(lines[9])
    ir_projector_base = float(lines[10])
    params_path = lines[11]
    return (
        ifshow,
        version,
        ir_path,
        color_path,
        depth_path,
        ir_width,
        ir_height,
        fx_rgb,
        color_ir_base,
        fx_ir,
        ir_projector_base,
        params_path,
    )


def _read_ir_raw(path: str, width: int, height: int) -> np.ndarray:
    """Load a 16‑bit raw IR image and return a normalised 8‑bit array.

    The original MATLAB code reads the raw IR image as ``uint16`` and
    immediately rescales it into an 8‑bit range.  The pixel values are
    normalised by dividing by the maximum value in the frame and
    multiplying by 255.  The resulting image is returned as ``uint8``.

    Parameters
    ----------
    path : str
        Path to the raw IR data file.
    width, height : int
        Dimensions of the image in pixels.

    Returns
    -------
    np.ndarray of shape (height, width), dtype=uint8

    Raises
    ------
    RuntimeError
        If the file cannot be read or does not contain the expected
        number of 16‑bit values.
    """
    expected_len = width * height
    try:
        data = np.fromfile(path, dtype=np.uint16, count=expected_len)
    except Exception as exc:
        raise RuntimeError(f"Failed to read IR raw data from '{path}': {exc}") from exc
    if data.size != expected_len:
        raise RuntimeError(
            f"IR raw file '{path}' does not contain the expected number of pixels "
            f"({expected_len}); found {data.size}."
        )
    # Reshape to (height, width).  MATLAB reads with dimensions [width, height]
    # and then transposes, which is equivalent to reading row‑major into a
    # (height, width) array in numpy.
    img = data.reshape((height, width)).astype(np.float32)
    # Normalise to 8‑bit
    max_val = float(img.max()) if img.size > 0 else 1.0
    if max_val <= 0.0:
        max_val = 1.0
    img_norm = (img * 255.0 / max_val).astype(np.uint8)
    return img_norm


def _gamma_correct(img: np.ndarray, threshold: float = 60.0, gamma: float = 0.5) -> np.ndarray:
    """Apply a simple gamma correction to an 8‑bit image when the mean is low.

    The MATLAB code performs a gamma correction with exponent 1/2 when the
    average brightness of the IR frame is below 60.  A lookup table is
    created to map each input intensity to an output intensity.

    Parameters
    ----------
    img : np.ndarray
        2‑D array of type ``uint8``.
    threshold : float, optional
        Mean intensity threshold below which gamma correction is applied.
    gamma : float, optional
        Exponent for the gamma correction.  The MATLAB code uses 1/2.

    Returns
    -------
    np.ndarray
        The gamma‑corrected image (or a copy of the input if no correction
        was applied).
    """
    mean_val = float(img.mean())
    if mean_val >= threshold:
        return img.copy()
    # Build lookup table.  The MATLAB implementation uses (i+0.5)/255
    # raised to the gamma power, multiplied by 255 and minus 0.5.  We
    # replicate that exactly and clip the result to [0,255].
    lut = np.empty(256, dtype=np.uint8)
    for i in range(256):
        f = ((i + 0.5) / 255.0) ** gamma
        val = int(f * 255.0 - 0.5)
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        lut[i] = val
    return lut[img]


def _shift_ir(img: np.ndarray, version: int, shift_x: int = 4, shift_y: int = 4) -> np.ndarray:
    """Apply a fixed pixel shift to the IR image.

    The IR and colour sensors of the camera module are displaced relative to
    one another.  The original MATLAB code accounts for this by copying
    pixels from a shifted region of the raw IR frame into a new array.
    Different hardware versions use different copy directions.

    For ``version == 0`` (a200) the IR frame is shifted down by ``shift_y``
    and left by ``shift_x``.  For ``version == 1`` (s300) it is shifted
    down by ``shift_y`` and right by ``shift_x``.  Pixels outside the
    shifted region are filled with zeros.

    Parameters
    ----------
    img : np.ndarray
        The 2‑D ``uint8`` IR image after normalisation and gamma correction.
    version : int
        Hardware version flag: 0 for a200 and 1 for s300.
    shift_x, shift_y : int, optional
        Pixel offsets in the x and y directions.

    Returns
    -------
    np.ndarray
        The shifted IR image.
    """
    h, w = img.shape
    out = np.zeros_like(img)
    if version == 0:
        # Copy a region starting shift_y rows below and ending at the bottom
        # into the upper part of the output, shifted right by shift_x pixels.
        max_i = h - shift_y
        if max_i > 0:
            out[:max_i, shift_x:] = img[shift_y:h, :w - shift_x]
    elif version == 1:
        max_i = h - shift_y
        if max_i > 0:
            out[:max_i, :w - shift_x] = img[shift_y:h, shift_x:w]
    else:
        # Unknown version: do not shift
        out = img.copy()
    return out


def _read_color_raw(path: str, width: int, height: int) -> np.ndarray:
    """Load a packed RGB raw image and return a 3‑channel uint8 array.

    The colour image is stored as a sequence of 8‑bit bytes arranged
    row‑wise.  Each pixel occupies three consecutive bytes (R, G, B).  The
    MATLAB code reads a [width*3, height] matrix and then reshapes it
    into an H×W×3 array by slicing every third element for each channel.

    Parameters
    ----------
    path : str
        Path to the raw colour data file.
    width, height : int
        Dimensions of the image in pixels.

    Returns
    -------
    np.ndarray of shape (height, width, 3), dtype=uint8

    Raises
    ------
    RuntimeError
        If the file cannot be read or does not contain the expected
        number of bytes.
    """
    expected_len = width * height * 3
    try:
        data = np.fromfile(path, dtype=np.uint8, count=expected_len)
    except Exception as exc:
        raise RuntimeError(f"Failed to read colour raw data from '{path}': {exc}") from exc
    if data.size != expected_len:
        raise RuntimeError(
            f"Colour raw file '{path}' does not contain the expected number of bytes "
            f"({expected_len}); found {data.size}."
        )
    # Reshape to (height, width*3)
    rows = data.reshape((height, width * 3))
    # Allocate output image
    cimg = np.empty((height, width, 3), dtype=np.uint8)
    # The MATLAB code assigns:
    # cimg(:,:,1) = obj_img2(:,1:3:ir_width*3);
    # cimg(:,:,2) = obj_img2(:,2:3:ir_width*3);
    # cimg(:,:,3) = obj_img2(:,3:3:ir_width*3);
    # Here we treat channel 0 as red, 1 as green, 2 as blue.  The raw file
    # stores pixels in RGB order.
    cimg[:, :, 0] = rows[:, 0::3]
    cimg[:, :, 1] = rows[:, 1::3]
    cimg[:, :, 2] = rows[:, 2::3]
    return cimg


def _read_depth_raw(path: str, width: int, height: int) -> np.ndarray:
    """Load a depth image from a 16‑bit raw file.

    Depth values are read as unsigned 16‑bit integers and reshaped into a
    (height, width) array.  No scaling or normalisation is applied.

    Parameters
    ----------
    path : str
        Path to the raw depth data file.
    width, height : int
        Dimensions of the image in pixels.

    Returns
    -------
    np.ndarray of shape (height, width), dtype=np.uint16
    """
    expected_len = width * height
    data = np.fromfile(path, dtype=np.uint16, count=expected_len)
    if data.size != expected_len:
        raise RuntimeError(
            f"Depth raw file '{path}' does not contain the expected number of pixels "
            f"({expected_len}); found {data.size}."
        )
    return data.reshape((height, width))


def _detect_corners(ir_img: np.ndarray, color_img: np.ndarray,
                    board_size: Tuple[int, int] = (8, 11)) -> Tuple[np.ndarray, np.ndarray]:
    """Detect chessboard corners in the IR and colour images.

    The detection function in ``sxCheckBoardDetect`` expects file paths, so
    temporary images are written to disk.  If detection fails on either
    image, a ``RuntimeError`` is raised.  The board size specifies the
    expected number of inner corners per row and column; both the given
    orientation and its transpose are attempted internally by
    ``detect_chessboard_corners``.

    Parameters
    ----------
    ir_img : np.ndarray
        2‑D array containing the (shifted, gamma‑corrected) IR image.
    color_img : np.ndarray
        2‑D array containing the greyscale colour image.
    board_size : tuple of int, optional
        Expected pattern size (cols, rows) of inner chessboard corners.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Arrays of shape (N, 2) containing the detected corner coordinates
        in the IR and colour images respectively.  The coordinates are
        floating point values in pixel units.

    Raises
    ------
    RuntimeError
        If the chessboard cannot be detected in either image.
    """
    # Write temporary images because the helper reads from disk
    with tempfile.TemporaryDirectory() as tmpdir:
        ir_path = os.path.join(tmpdir, "ir_tmp.png")
        color_path = os.path.join(tmpdir, "color_tmp.png")
        # Save as PNG to preserve pixel values
        cv2.imwrite(ir_path, ir_img)
        cv2.imwrite(color_path, color_img)
        # Detect in colour image first
        color_corners, _ = detect_chessboard_corners(color_path, board_size=board_size)
        if color_corners is None:
            raise RuntimeError("Failed to detect chessboard in colour image.")
        ir_corners, _ = detect_chessboard_corners(ir_path, board_size=board_size)
        if ir_corners is None:
            raise RuntimeError("Failed to detect chessboard in IR image.")
        return ir_corners, color_corners


def _compute_depth_vector(ir_corners: np.ndarray, depth_img: np.ndarray,
                          fx_rgb: float, color_ir_base: float,
                          unit_scaler: float = 0.003) -> np.ndarray:
    """Compute the disparity term for each chessboard corner.

    In the original MATLAB code a depth‑dependent term ``depthV`` is
    subtracted from the horizontal coordinate difference when fitting the
    polynomial warp.  The term is computed for each corner as::

        depthV(i) = (fx_rgb * color_ir_base / unit_scaler) / depth_img(y_i, x_i)

    where ``x_i`` and ``y_i`` are rounded IR corner coordinates.  Points
    with zero or negative depth are considered invalid and raise an
    exception.

    Parameters
    ----------
    ir_corners : np.ndarray
        N×2 array of corner positions (x, y) in the IR image.
    depth_img : np.ndarray
        2‑D array of raw depth values, same resolution as the IR image.
    fx_rgb : float
        Focal length of the colour camera (pre‑scaled by 0.5).
    color_ir_base : float
        Baseline between the colour and IR cameras in millimetres.
    unit_scaler : float, optional
        The constant ``0.003`` from the MATLAB code.  Converts depth units
        into metres if the depth values are in millimetres.

    Returns
    -------
    np.ndarray
        1‑D array ``depthV`` of length equal to the number of corners.

    Raises
    ------
    RuntimeError
        If a depth value is zero or negative at any corner position.
    """
    n_points = ir_corners.shape[0]
    depthV = np.empty(n_points, dtype=np.float64)
    # Precompute the scale factor
    scale = fx_rgb * color_ir_base / unit_scaler
    # Round corner coordinates to the nearest integer indices
    x_idx = np.rint(ir_corners[:, 0]).astype(np.int64)
    y_idx = np.rint(ir_corners[:, 1]).astype(np.int64)
    h, w = depth_img.shape
    for i in range(n_points):
        x = x_idx[i]
        y = y_idx[i]
        # Clip indices into image bounds
        if x < 0:
            x = 0
        elif x >= w:
            x = w - 1
        if y < 0:
            y = 0
        elif y >= h:
            y = h - 1
        d = float(depth_img[y, x])
        if d <= 0.0:
            raise RuntimeError(
                f"Depth image contains zero or negative value at IR corner index {i}."
            )
        depthV[i] = scale / d
    return depthV


def _fit_polynomial(ir_corners: np.ndarray, color_corners: np.ndarray,
                    depthV: np.ndarray, fx_rgb: float, color_ir_base: float,
                    fx_ir: float, ir_projector_base: float,
                    version: int) -> np.ndarray:
    """Solve for the polynomial warp parameters.

    This function reproduces the least‑squares fitting and parameter
    post‑processing from the MATLAB code.  It constructs a design matrix
    with quadratic, cross and linear terms of the IR corner coordinates and
    fits separate models for the x and y directions.  The depth term is
    subtracted from the horizontal coordinate difference when fitting the
    x direction.  A constant ``beta`` is also computed, although it is not
    directly used in the final x parameter fit here.  After fitting, the
    coefficients are combined and scaled by ``2**20`` to produce the
    13‑element integer parameter vector required by the downstream
    hardware.

    Parameters
    ----------
    ir_corners : np.ndarray
        N×2 array of IR corner coordinates.
    color_corners : np.ndarray
        N×2 array of colour corner coordinates.
    depthV : np.ndarray
        1‑D array of depth disparity terms, same length as the number of corners.
    fx_rgb, color_ir_base, fx_ir, ir_projector_base : float
        Camera and baseline parameters from the configuration.
    version : int
        Hardware version flag: 0 for a200, 1 for s300.

    Returns
    -------
    np.ndarray
        Array of 13 integer parameters scaled by 2^20.
    """
    # Extract coordinate vectors
    ir_px = ir_corners[:, 0].astype(np.float64)
    ir_py = ir_corners[:, 1].astype(np.float64)
    color_px = color_corners[:, 0].astype(np.float64)
    color_py = color_corners[:, 1].astype(np.float64)

    # Compute beta (unused directly in fitting of x_params but needed for output)
    beta = fx_rgb * color_ir_base / (fx_ir * ir_projector_base)

    # Design matrices for quadratic polynomial: ax*x^2 + bx*y^2 + cx*xy + dx*x + ex*y + fx
    ones = np.ones_like(ir_px)
    mat_p = np.column_stack(
        [ir_px ** 2, ir_py ** 2, ir_px * ir_py, ir_px, ir_py, ones]
    )

    # Fit x direction.  The MATLAB code subtracts the depth term from the
    # colour minus IR difference.  We solve for six coefficients: ax..f.
    rhs_x = (color_px - ir_px - depthV)
    x_params, *_ = np.linalg.lstsq(mat_p, rhs_x, rcond=None)

    # Fit y direction.  No depth correction is applied here.
    rhs_y = (color_py - ir_py)
    y_params, *_ = np.linalg.lstsq(mat_p, rhs_y, rcond=None)

    # Post‑fit adjustment analogous to the MATLAB code.  In MATLAB the
    # constant term of x_params (index 5) is offset by fx_rgb*color_ir_base/(0.003*1200).
    # This term accounts for a nominal disparity at a 1.2 m distance.
    x_params = x_params.copy()
    x_params[5] += fx_rgb * color_ir_base / 0.003 / 1200.0

    # For version==0 (a200) the MATLAB code doubles beta for output; version==1 leaves it.
    beta_out = beta * (2.0 if version == 0 else 1.0)

    # Combine parameters into the 13‑element vector according to the original ordering
    ax, bx, cx, dx, ex, fx0 = x_params  # dx and ex correspond to linear terms
    ay, by, cy, dy, ey, fy0 = y_params
    dx_start_row = fx0
    dxdx_start_row = ax + dx  # ax + dx
    dydx_start = bx + ex       # bx + ex
    dy_start_row = fy0
    dxdy_start_row = ay + dy   # ay + dy
    dydy_start = by + ey       # by + ey
    params = np.array([
        ax,
        bx,
        cx,
        ay,
        by,
        cy,
        dx_start_row,
        dxdx_start_row,
        dydx_start,
        dy_start_row,
        dxdy_start_row,
        dydy_start,
        beta_out,
    ], dtype=np.float64)
    # Scale and floor to integers as in MATLAB
    inc_params = np.floor(params * (2.0 ** 20)).astype(np.int64)
    return inc_params


def _write_params(params: np.ndarray, version: int, out_path: str) -> None:
    """Write the integer parameters to a file.

    The MATLAB routine chooses two different output formats depending on
    ``version``.  For version 1 (s300) each parameter is written on its
    own line.  For version 0 (a200) the parameters are wrapped in curly
    braces and separated by commas for compatibility with certain
    firmware parsers.

    Parameters
    ----------
    params : np.ndarray
        1‑D array of 13 integer parameters.
    version : int
        Hardware version flag.
    out_path : str
        Destination path for the parameter file.  Intermediate
        directories will be created if necessary.
    """
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if version == 1:
            for val in params:
                f.write(f"{int(val)}\n")
        else:
            # version 0: wrap in curly braces and separate with commas
            joined = ",\n ".join(str(int(v)) for v in params)
            f.write("{" + joined + "\n}")


def hor_reg_auto(config_path: str,
                 board_size: Tuple[int, int] = (8, 11)) -> Dict[str, Any]:
    """Perform IR–colour registration using the provided configuration file.

    This is the top‑level function that orchestrates reading inputs,
    performing image pre‑processing, detecting calibration target corners,
    fitting the warping polynomial and writing the calibration parameters.
    The steps mirror those in the original MATLAB implementation.

    This function is designed for production use - it contains NO visualization
    or display code.  All computation is pure and suitable for headless
    environments and automated processing pipelines.

    Parameters
    ----------
    config_path : str
        Path to the text file containing the registration configuration.
    board_size : tuple of int, optional
        Expected number of inner corners per chessboard row and column.  It
        defaults to (8, 11) as used in many calibration boards.  The
        detection function will automatically try both orientations.

    Returns
    -------
    dict
        A dictionary containing all computed results for potential use by
        visualization or analysis tools:
        
        - 'ir_corners': IR corner coordinates (N×2 array)
        - 'color_corners': Color corner coordinates (N×2 array)
        - 'depthV': Depth disparity terms (1D array)
        - 'params': Final integer parameters (1D array)
        - 'config': Configuration parameters (dict)
        - 'images': Processed images (dict with 'ir_shifted', 'color_gray', 'depth_img')
        - 'error_stats': RMS reprojection errors (dict)

    Raises
    ------
    RuntimeError
        If any of the inputs cannot be read or if the chessboard cannot
        be detected.
    """
    (
        ifshow,
        version,
        ir_path,
        color_path,
        depth_path,
        ir_width,
        ir_height,
        fx_rgb,
        color_ir_base,
        fx_ir,
        ir_projector_base,
        params_path,
    ) = _read_config(config_path)

    # Load and pre‑process IR image
    ir_raw = _read_ir_raw(ir_path, ir_width, ir_height)
    ir_gamma = _gamma_correct(ir_raw)
    ir_shifted = _shift_ir(ir_gamma, version)

    # Load colour raw image and convert to greyscale
    color_raw = _read_color_raw(color_path, ir_width, ir_height)
    # Convert to greyscale using OpenCV (handles channel ordering)
    # The packed colour raw is assembled as RGB in _read_color_raw.  Use
    # COLOR_RGB2GRAY to obtain a luminance image.
    color_gray = cv2.cvtColor(color_raw, cv2.COLOR_RGB2GRAY)

    # Load depth image
    depth_img = _read_depth_raw(depth_path, ir_width, ir_height)

    # Detect chessboard corners
    ir_corners, color_corners = _detect_corners(ir_shifted, color_gray, board_size)
    # Ensure the same number of corners were found in both images
    if ir_corners.shape[0] != color_corners.shape[0]:
        raise RuntimeError(
            f"Mismatch in number of detected corners: IR has {ir_corners.shape[0]}, "
            f"colour has {color_corners.shape[0]}."
        )

    # Use all detected points by default.  The original MATLAB code used
    # only (rows-1)*(cols-1) points for certain boards.  If the user
    # wishes to replicate that behaviour exactly they may slice the arrays
    # here.  For example:
    #   rows, cols = board_size
    #   sizePattern = (rows - 1) * (cols - 1)
    #   ir_corners = ir_corners[:sizePattern]
    #   color_corners = color_corners[:sizePattern]

    # Compute depth disparity term
    depthV = _compute_depth_vector(
        ir_corners, depth_img, fx_rgb=fx_rgb, color_ir_base=color_ir_base
    )

    # Fit polynomial coefficients
    inc_params = _fit_polynomial(
        ir_corners,
        color_corners,
        depthV,
        fx_rgb=fx_rgb,
        color_ir_base=color_ir_base,
        fx_ir=fx_ir,
        ir_projector_base=ir_projector_base,
        version=version,
    )

    # Write the parameters to file
    _write_params(inc_params, version, params_path)

    # Calculate RMS reprojection errors for quality assessment
    ir_px = ir_corners[:, 0].astype(np.float64)
    ir_py = ir_corners[:, 1].astype(np.float64)
    color_px = color_corners[:, 0].astype(np.float64)
    color_py = color_corners[:, 1].astype(np.float64)
    
    # Design matrix for prediction
    ones = np.ones_like(ir_px)
    mat_p = np.column_stack(
        [ir_px ** 2, ir_py ** 2, ir_px * ir_py, ir_px, ir_py, ones]
    )
    
    # Extract parameters for prediction
    ax, bx, cx, ay, by, cy, dx_start_row, dxdx_start_row, dydx_start, dy_start_row, dxdy_start_row, dydy_start, beta_out = inc_params / (2.0 ** 20)
    
    # Predict transformed coordinates
    pred_x = ir_px + mat_p @ np.array([ax, bx, cx, 0, 0, dx_start_row]) + depthV
    pred_y = ir_py + mat_p @ np.array([ay, by, cy, 0, 0, dy_start_row])
    
    # Calculate RMS errors
    rms_x = np.sqrt(np.mean((pred_x - color_px) ** 2))
    rms_y = np.sqrt(np.mean((pred_y - color_py) ** 2))

    # Return comprehensive results dictionary
    results = {
        'ir_corners': ir_corners,
        'color_corners': color_corners,
        'depthV': depthV,
        'params': inc_params,
        'config': {
            'version': version,
            'fx_rgb': fx_rgb,
            'color_ir_base': color_ir_base,
            'fx_ir': fx_ir,
            'ir_projector_base': ir_projector_base,
            'board_size': board_size,
            'params_path': params_path
        },
        'images': {
            'ir_shifted': ir_shifted,
            'color_gray': color_gray,
            'depth_img': depth_img
        },
        'error_stats': {
            'rms_x': rms_x,
            'rms_y': rms_y,
            'total_corners': len(ir_corners)
        }
    }

    return results


# Backward compatibility - keep the old function name for existing scripts
def hor_reg_auto_V1(config_path: str) -> None:
    """Backward compatibility wrapper for hor_reg_auto."""
    results = hor_reg_auto(config_path)
    # Print basic results for backward compatibility
    print(f"RMS reprojection error: {results['error_stats']['rms_x']:.3f} px (x), {results['error_stats']['rms_y']:.3f} px (y)")
    print(f"[OK] params saved to {results['config']['params_path']}")


# Allow script-style execution
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stereo IR‑Color registration (auto)")
    ap.add_argument("config", help="path to *.txt config (see hor_reg_sk500.txt)")
    ap.add_argument("--board-size", nargs=2, type=int, default=[8, 11], 
                   help="chessboard size (cols rows), default: 8 11")
    args = ap.parse_args()
    
    # Run registration and print basic results
    results = hor_reg_auto(args.config, board_size=tuple(args.board_size))
    print(f"RMS reprojection error: {results['error_stats']['rms_x']:.3f} px (x), {results['error_stats']['rms_y']:.3f} px (y)")
    print(f"[OK] params saved to {results['config']['params_path']}") 