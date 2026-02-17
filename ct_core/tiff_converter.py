# Description: This script takes vff file data and saves it as TIFF images
# Written by Falk Wiegmann at the University of British Columbia in May 2024.

import os
import numpy as np
import imageio
from tqdm import tqdm

def save_vff_to_tiff(vff_data, target_directory=None, filename=None, verbose=True, compute_average_img=True):
    '''
    This function takes the data from a VFF file and saves it as TIFF images.
    :param vff_data: The 3D numpy array with the voxel data
    :param target_directory: The directory where the TIFF images should be saved
    :param filename: Optional filename for 2D data
    :param verbose: Boolean to print out the progress
    :param compute_average_img: Boolean to compute the average image and save it as well
    :return: None
    '''
    # Default target directory
    if target_directory is None:
        target_directory = os.path.join(os.getcwd(), 'TIFF_output')

    # Create the target directory if it does not exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Create the TIFF files
    if vff_data.ndim == 3:
        slices = tqdm(range(len(vff_data)), desc='Saving TIFF slices',
                      unit='slice', disable=not verbose)
        for slice_index in slices:
            imageio.imwrite(os.path.join(target_directory, f'slice_{slice_index}.tiff'), vff_data[slice_index])

    elif vff_data.ndim == 2:
        # Save the VFF data as a TIFF image
        if filename is not None:
            # Save the VFF data as a TIFF image with the specified filename
            imageio.imwrite(os.path.join(target_directory, f'{filename}.tiff'), vff_data)
        else:
            # Save the VFF data as a TIFF image with a default name
            imageio.imwrite(os.path.join(target_directory, 'image.tiff'), vff_data)

        if verbose:
            print("Saved the image as a TIFF file")

    else:
        raise ValueError("The input data must be a 2D or 3D numpy array")

    if compute_average_img and vff_data.ndim == 3:
        # Save the average image as a TIFF file
        imageio.imwrite(os.path.join(target_directory, 'averaged.tiff'), np.average(vff_data, axis=0))

        if verbose:
            print("Saved the average image as a TIFF file")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Convert a VFF file to TIFF slices')
    p.add_argument('input', help='Path to input VFF file')
    p.add_argument('--outdir', default=None,
                   help='Output directory for TIFF slices (default: input path without extension)')
    p.add_argument('--average', action='store_true',
                   help='Also save an averaged projection image (for 3D volumes)')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-slice progress output')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    try:
        from .vff_io import read_vff
    except ImportError:
        from vff_io import read_vff
    header, data = read_vff(filename=args.input, verbose=not args.quiet)

    outdir = args.outdir
    if outdir is None:
        # Strip extension: foo.vff -> foo/
        outdir = os.path.splitext(args.input)[0]

    save_vff_to_tiff(data, target_directory=outdir,
                     verbose=not args.quiet, compute_average_img=args.average)
