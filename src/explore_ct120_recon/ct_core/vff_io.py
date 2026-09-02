# Description: This script reads a VFF file and returns the header and the data.
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import numpy as np
import os
import tempfile
from pathlib import Path
import xmltodict
import torch

# key insights into what a vff file looks like came from https://imagej.net/ij/plugins/download/Multi_VFF_Opener.java

def read_vff_header(filename, verbose=True):
    '''
    This function reads the header of a VFF file and returns a dictionary with the header data.
    Don't call this on its own (use read_vff() instead).
    :param filename: The path to the VFF file (including the file name.vff)
    :return: A dictionary with the header data
    '''
    header = {}
    with open(filename, 'rb') as file:
        # Using 'latin-1' to avoid decoding issues
        content = file.read(1000).decode('latin-1') # Read the first 1000 bytes (header is < 1000 bytes)

        # Split on form feed, take the first part
        try:
            header_content, _ = content.split('\f', 1)
        except ValueError: # sometimes there is no form feed
            header_content = content
        lines = header_content.splitlines()
        for line in lines:
            if '=' in line:  # Simple check to filter out irrelevant lines
                key, value = line.strip().split('=')
                header[key.strip()] = value.strip()[:-1]

        # Print the header data if verbose is set to True
        if verbose:
            print("--------------------------Header Data----------------------------")
            for key, value in header.items():
                print(f"{key}: {value}")
            print("-----------------------End of Header Data-------------------------")

    return header

def read_vff_data(filename, header, verbose=True):
    '''
    This function reads the data of a VFF file and returns a 3D numpy array.
    Don't call this on its own (use read_vff() instead).
    :param filename: The path to the VFF file (including the file name.vff)
    :param header: The header data of the VFF file (to get the dimensions and data type)
    :return: A 3D numpy array with the voxel data
    '''
    SYSTEM_MEMORY = 0.3 # in GB, change if larger arrays should be loaded into memory instead of using memmap

    xdim = int(header['size'].split()[0])
    ydim = int(header['size'].split()[1])
    if len(header['size'].split()) == 2:
        zdim = 1
    else:
        zdim = int(header['size'].split()[2])
    bits = int(header['bits'])

    # Assuming 8 or 16 bits per voxel
    data_type = np.dtype('>b') if bits == 8 else np.dtype('>h')
    data_size = xdim * ydim * zdim * int(bits/8)

    if data_size > SYSTEM_MEMORY * 1024**3:
        if verbose:
            print(f"Data size is {data_size/1024**3:.3f} GB, which is larger than the specified system memory allocation of {SYSTEM_MEMORY} GB. Loading data as memory-mapped file.")
        data = np.memmap(filename, dtype=data_type, mode='c', offset=os.path.getsize(filename)-data_size, shape=(zdim, ydim, xdim))

    else:
        data = np.fromfile(filename, dtype=data_type, offset=os.path.getsize(filename)-data_size).reshape(zdim, ydim, xdim)

    return data

def read_vff(filename, verbose=True):
    '''
    This function reads a VFF file and returns the header and the data.
    This is the main function that should be called.
    Set verbose to False to suppress the print output of the header.
    :return: header: A dictionary with the header data,
             data: A 3D numpy array with the voxel data (z, y, x)
    '''

    header = read_vff_header(filename, verbose=verbose)
    data = read_vff_data(filename, header, verbose=verbose)

    if verbose:
        print("Data loaded successfully, data shape:", data.shape)

    return header, data

def write_vff(filename, header, data, verbose=True):
    """
    Write a VFF file from a header dict and 2D/3D NumPy array.

    The ASCII header is written in GE's `ncaa` dialect so the output loads
    directly in external software (Amalytics / MicroView) — the magic word
    `ncaa`, the GE keywords (rank/type/bands/format=slice), and voxel size /
    HU-calibration fields are all present. The binary payload is unchanged:
    big-endian, z-major, form-feed separated, exactly as GE expects.

    Voxel data is assumed to already be in Hounsfield Units (air floor ≈ -1000).
    We write water=0 / air=-1000 so Amalytics' HU formula
    HU = 1000*(v-water)/(water-air) is the identity — the displayed value equals
    the stored value whether or not the tool applies its calibration.

    :param filename: Path to output .vff file
    :param header: Dict with metadata keys; recognized keys:
        'bits' (8 or 16, default 16), 'elementsize' (isotropic voxel size in mm)
        or 'spacing' ("dx dy dz" — first token used as elementsize),
        'water'/'air' (HU calibration anchors, default 0 / -1000).
        Size is always inferred from `data`.
    :param data: 3D NumPy array shaped (z, y, x) or 2D array (y, x)
    """
    # Convert input to NumPy array and ensure 3D shape
    arr = np.array(data, copy=False)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Data must be 2D or 3D array, got {arr.ndim}D")
    zdim, ydim, xdim = arr.shape

    # GE `ncaa` uses signed-short slices; pick dtype from bits
    bits = int(header.get('bits', 16))
    dtype = np.dtype('>u1') if bits == 8 else np.dtype('>i2') if bits == 16 else None
    if dtype is None:
        raise ValueError("Unsupported bits per voxel: must be 8 or 16")

    # Voxel size (mm). GE's elementsize is a single scalar, so anisotropic
    # volumes cannot be expressed faithfully — use the in-plane size and warn.
    if 'elementsize' in header:
        elementsize = float(header['elementsize'])
    elif 'spacing' in header:
        toks = str(header['spacing']).split()
        elementsize = float(toks[0])
        if verbose and len(toks) >= 3 and not all(abs(float(t) - elementsize) < 1e-9 for t in toks):
            print(f"  WARNING: anisotropic spacing {toks} — GE elementsize is scalar; "
                  f"writing {elementsize} for all axes.")
    else:
        elementsize = 1.0

    # HU calibration anchors — identity mapping by default (data already in HU)
    water = float(header.get('water', 0.0))
    air = float(header.get('air', -1000.0))

    # Ensure big-endian C-contiguous data.
    # Float input (HU volumes are float32) must be ROUNDED and CLIPPED, not
    # `astype`d: astype truncates toward zero (-999.7 HU -> -999, a systematic
    # +0.5 HU bias on the air floor) and, worse, WRAPS on overflow, so a single
    # +40000 HU voxel would silently store as -25536.
    if arr.dtype != dtype or arr.dtype.byteorder != '>' or not arr.flags['C_CONTIGUOUS']:
        if np.issubdtype(arr.dtype, np.floating):
            info = np.iinfo(dtype)
            n_clip = int(np.count_nonzero((arr < info.min) | (arr > info.max)))
            if n_clip and verbose:
                print(f"  WARNING: {n_clip} voxel(s) outside the {bits}-bit range "
                      f"[{info.min}, {info.max}] — clipped.")
            arr_be = np.rint(np.clip(arr, info.min, info.max)).astype(dtype)
        else:
            arr_be = arr.astype(dtype)
    else:
        arr_be = arr

    # Data range (from the stored integer values) for the header min/max fields
    dmin = float(arr_be.min()) if arr_be.size else 0.0
    dmax = float(arr_be.max()) if arr_be.size else 0.0

    # Build GE `ncaa` header (keys/format mirror working Green-*.vff files)
    lines = [
        "ncaa",
        "rank=3;",
        "type=raster;",
        "modality=CT;",
        f"size={xdim} {ydim} {zdim};",
        "origin=0.0 0.0 0.0;",
        "bands=1;",
        f"bits={bits};",
        "format=slice;",
        "spacing=1.00 1.00 1.00;",
        f"elementsize={elementsize:.6f};",
        f"min={dmin:.6f};",
        f"max={dmax:.6f};",
        f"water={water:.6f};",
        f"air={air:.6f};",
    ]
    header_text = "\n".join(lines) + "\n\f\n"

    if verbose:
        print(f"Writing VFF to {filename}: shape={arr_be.shape}, dtype={arr_be.dtype}, "
              f"HU[min={dmin:.0f}, max={dmax:.0f}]")

    with open(filename, 'wb') as f:
        f.write(header_text.encode('latin-1'))

        arr_be.tofile(f)


class VFFDataset:
    def __init__(self, folder: str, xml_file: str, save_headers: bool = False, tensor_projections: bool = False,
                 paths_str: str = "acq*", exclude_pred_paths: bool = False, projection_spacing = None,
                 sub_scan='-00-'):
        self.folder = folder
        self.paths = sorted(Path(folder).glob(paths_str))
        if exclude_pred_paths is True:
            self.paths = [p for p in self.paths if not p.name.endswith("_pred.vff")]

        # Apply sub-scan filter only to acquisition files (not sequential proj-* files)
        if sub_scan:
            self.paths = [p for p in self.paths
                          if p.name.startswith('proj-') or sub_scan in str(p)]

        with open(xml_file, 'r') as f:
            header = xmltodict.parse(f.read())
        sp = header['Series']['SeriesParams']
        self.num_projections = len(self.paths) #int(sp['ViewCount'])
        if projection_spacing is not None:
            self.imaging_angle = projection_spacing * self.num_projections
            print(f"Using custom projection spacing: {projection_spacing} degrees per projection, total angle: {self.imaging_angle:.2f} degrees")
        else:
            self.imaging_angle = float(sp['IncrementAngle']) * self.num_projections
            print(f"Using default projection spacing: {sp['IncrementAngle']} degrees per projection, total angle: {self.imaging_angle:.2f} degrees")

        self.starting_angle_offset = float(header['Series']['AngleOffset']) + 120

        # Read first frame for shape/dtype
        hdr, data = self._read_vff(self.paths[0])
        data = data.squeeze(0)
        self.det_rows, self.det_cols = data.shape

        # Create memmap for projections. Use a unique per-process temp file
        # (SLURM_TMPDIR if set, else the system temp dir) rather than a fixed
        # path inside the scan folder: mode='w+' truncates+recreates the file
        # on open, so a fixed shared path caused concurrent reconstructions of
        # the same scan to stomp on each other's memmap (Bus errors / "mmap
        # length is greater than file size" when one process's re-open
        # invalidated another's in-progress mapping).
        shape = (len(self.paths), self.det_rows, self.det_cols)
        tmp_dir = os.environ.get('SLURM_TMPDIR') or tempfile.gettempdir()
        memmap_fd, memmap_path = tempfile.mkstemp(
            suffix='.dat', prefix='detector_values_', dir=tmp_dir)
        os.close(memmap_fd)
        self._memmap_path = memmap_path
        self.projections = np.memmap(memmap_path,
                                     dtype=data.dtype,
                                     mode='w+',
                                     shape=shape)
        self.projection_angles = np.zeros(len(self.paths), dtype=np.float32)

        if save_headers:
            self.headers = []

        # Fill memmap and angles
        for idx, p in enumerate(self.paths):
            hdr, dat = self._read_vff(p)
            arr = dat.squeeze(0)
            # Reverse byteswap to original order before writing
            original = arr.byteswap().view(arr.dtype.newbyteorder())
            self.projections[idx] = original
            self.projection_angles[idx] = float(hdr['gantryPosition'])

            if save_headers:
                self.headers.append(hdr)

        # Projections are now fully loaded into the memmap; the backing file
        # itself is scratch and can be removed immediately (the OS keeps the
        # already-mmap'd data accessible via the existing file descriptor
        # until this process exits or self.projections is released).
        os.remove(self._memmap_path)

        # Convert memmap to native byte order so torch can load it
        if self.projections.dtype.byteorder not in ('=', '<'):  # not native little endian
            # numpy 2.0 dropped ndarray.newbyteorder(), use view with new dtype
            new_dtype = self.projections.dtype.newbyteorder()
            self.projections = self.projections.byteswap().view(new_dtype)

        # Convert projections to float32 tensor if requested
        if tensor_projections:
            self.projections = torch.from_numpy(np.asarray(self.projections, dtype=np.float32))

        # Prepare angles in radians — prefer actual gantry positions over synthetic linspace
        gantry = torch.from_numpy(self.projection_angles)
        gantry_range = float(gantry.max() - gantry.min())
        if gantry_range > 1.0:
            # Gantry positions decrease during acquisition; invert to get increasing angles
            relative = gantry[0] - gantry
            angles_deg = (self.starting_angle_offset + relative) % 360
            print(f"Using actual gantry angles (range {gantry_range:.2f}°)")
        else:
            # Fallback: synthetic angles with arange (not linspace — avoids off-by-one)
            step = self.imaging_angle / self.num_projections
            angles_deg = (torch.arange(self.num_projections, dtype=torch.float32) * step
                          + self.starting_angle_offset) % 360
            print(f"Gantry angles unavailable, using synthetic arange (step={step:.4f}°)")
        self.angles_rad = torch.deg2rad(angles_deg)

    def _read_vff(self, path: Path):
        # Use same header parsing as standalone functions
        header = {}
        with open(path, 'rb') as f:
            raw = f.read(1000).decode('latin-1', errors='ignore')
        try:
            header_content, _ = raw.split('\f', 1)
        except ValueError:
            header_content = raw
        for line in header_content.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip().rstrip(',;')

        # Determine dtype from header
        bits = int(header.get('bits', '16'))
        dtype = np.dtype('>b' if bits == 8 else '>h')
        dims = [int(x) for x in header['size'].split()]
        if len(dims) == 2:
            z, y, x = 1, dims[1], dims[0]
        else:
            z, y, x = dims[2], dims[1], dims[0]
        total_bytes = x * y * z * bits // 8
        offset = os.path.getsize(path) - total_bytes
        data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(z, y, x))
        return header, data


if __name__ == '__main__':
    read_vff(filename = "/Users/falk/Downloads/Shelley phantom full scan 75um 16ms.vff", verbose=True)


def detector_serial_from_scan(scan_folder) -> "str | None":
    """Read ``serialNumber`` from any acquisition frame's VFF header.

    Lives here (rather than in muNeRF's detector_warp) so the geometry
    auto-calibration in ct_core/detector_psi.py can key its cache by detector
    without importing anything from muNeRF — ct_core is the SHARED layer, and
    the dependency must not point the other way.
    """
    from pathlib import Path as _P
    scan_folder = _P(scan_folder)
    for pattern in ("acq-*.vff", "proj-*.vff"):
        frames = sorted(scan_folder.glob(pattern))
        if frames:
            try:
                head = frames[0].open("rb").read(2048).replace(b"\x00", b"")
                for line in head.decode("latin-1").splitlines():
                    if line.startswith("serialNumber="):
                        return line.split("=", 1)[1].rstrip(";").strip()
            except OSError:
                return None
    return None
