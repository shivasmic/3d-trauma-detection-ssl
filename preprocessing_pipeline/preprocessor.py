"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import os
import glob
import pydicom
import numpy as np
import pandas as pd
import scipy.ndimage
import nibabel as nib
from dotenv import load_dotenv
from log import get_preprocessing_logger
from scipy.ndimage import label as nd_label
from multiprocessing import Pool
from functools import partial
import time

load_dotenv()

CONFIG = {
    "TARGET_SPACING_MM": tuple(map(float, os.getenv("TARGET_SPACING_MM").split(","))),
    "INPUT_DIMENSIONS": tuple(map(int, os.getenv("INPUT_DIMENSIONS").split(","))),
    "HU_CLIP_RANGE": tuple(map(int, os.getenv("HU_CLIP_RANGE").split(","))),
    "TRAIN_IMAGES_DIR": os.getenv("TRAIN_IMAGES_DIR"),
    "SEGMENTATIONS_DIR": os.getenv("SEGMENTATIONS_DIR"),
    "OUTPUT_DIR": os.getenv("OUTPUT_DIR"),
    "LABELS_CSV": os.getenv("LABELS_CSV"),
}

logger = get_preprocessing_logger()


def load_dicom_series(series_path):
    """Load DICOM series and return volume + spacing."""
    dicom_files = glob.glob(os.path.join(series_path, "*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {series_path}")
    
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(slices[0].SliceThickness)
    except AttributeError:
        logger.warning(f"Missing spacing tags in {series_path}. Using defaults.")
        pixel_spacing = [1.0, 1.0]
        slice_thickness = 1.0
    
    current_spacing = np.array([slice_thickness] + list(pixel_spacing), dtype=np.float32)
    
    image = np.stack([s.pixel_array for s in slices], axis=0).astype(np.int16)
    
    rescale_slope = slices[0].get('RescaleSlope', 1)
    rescale_intercept = slices[0].get('RescaleIntercept', 0)
    image = image * rescale_slope + rescale_intercept
    
    return image, current_spacing


def resample_and_normalize(image, current_spacing, config):
    """Resample to target spacing and normalize to [0, 1]."""
    
    target_spacing = np.array(config['TARGET_SPACING_MM'])
    resize_factor = current_spacing / target_spacing
    
    new_shape = np.round(image.shape * resize_factor).astype(int)
    actual_resize_factor = new_shape / image.shape
    
    image_resampled = scipy.ndimage.zoom(image, actual_resize_factor, order=3)
    
    image_clipped = np.clip(image_resampled, config['HU_CLIP_RANGE'][0], config['HU_CLIP_RANGE'][1])
    min_val = image_clipped.min()
    max_val = image_clipped.max()
    image_normalized = (image_clipped - min_val) / (max_val - min_val)
    
    return image_normalized


def process_nii_mask(nii_path, series_original_shape, series_resampled_shape):
    """Load, align, and resample mask to match DICOM volume."""
    
    nii_mask = nib.load(nii_path)
    mask_data = np.round(nii_mask.get_fdata()).astype(np.uint8)
    
    if mask_data.shape != series_original_shape:
        if mask_data.shape == (series_original_shape[2], series_original_shape[1], series_original_shape[0]):
            mask_data = np.transpose(mask_data, (2, 1, 0))
        elif mask_data.shape == (series_original_shape[1], series_original_shape[2], series_original_shape[0]):
            mask_data = np.transpose(mask_data, (2, 0, 1))
    
    resize_factor = np.array(series_resampled_shape) / np.array(mask_data.shape)
    mask_resampled = scipy.ndimage.zoom(mask_data, resize_factor, order=0).astype(np.uint8)
    mask_resampled = np.clip(mask_resampled, 0, 5)
    
    return mask_resampled


def center_crop_or_pad(volume, final_shape):
    """CENTER crop/pad to fixed size (NOT liver-centered!)."""
    
    pad_z = max(0, final_shape[0] - volume.shape[0])
    pad_y = max(0, final_shape[1] - volume.shape[1])
    pad_x = max(0, final_shape[2] - volume.shape[2])
    
    pad_before = np.array([pad_z // 2, pad_y // 2, pad_x // 2])
    
    if any([pad_z > 0, pad_y > 0, pad_x > 0]):
        padding = [(pad_z // 2, pad_z - pad_z // 2),
                  (pad_y // 2, pad_y - pad_y // 2),
                  (pad_x // 2, pad_x - pad_x // 2)]
        volume = np.pad(volume, padding, mode='constant', constant_values=0)
    
    start_z = (volume.shape[0] - final_shape[0]) // 2
    start_y = (volume.shape[1] - final_shape[1]) // 2
    start_x = (volume.shape[2] - final_shape[2]) // 2
    
    crop_start = np.array([start_z, start_y, start_x])
    offsets = crop_start - pad_before
    
    volume_cropped = volume[
        start_z : start_z + final_shape[0],
        start_y : start_y + final_shape[1],
        start_x : start_x + final_shape[2]
    ]
    
    return volume_cropped, offsets


def process_series(series_path, series_id, config, injury_df):
    """Process a single DICOM series with liver injury."""
    
    try:
        series_id_int = int(series_id)
    except ValueError:
        logger.error(f"Invalid series_id: {series_id}")
        return
    
    series_row = injury_df[injury_df['series_id'] == series_id_int]
    
    if series_row.empty:
        return
    
    has_liver_injury = series_row['has_liver'].values[0]
    
    if has_liver_injury == 0:
        return
    
    logger.info(f"Processing Series: {series_id} *** LIVER INJURY ***")
    
    output_file = os.path.join(config['OUTPUT_DIR'], f"{series_id}_liver.npz")
    if os.path.exists(output_file):
        logger.info(f"  ⏭️  Skipping {series_id}: Already processed")
        return
    
    try:
        raw_image, current_spacing = load_dicom_series(series_path)
    except Exception as e:
        logger.error(f"Skipping series {series_id}: DICOM loading error: {e}")
        return
    
    original_shape = raw_image.shape
    logger.info(f"  Original DICOM shape: {original_shape}")
    
    volume_resampled = resample_and_normalize(raw_image, current_spacing, config)
    resampled_shape = volume_resampled.shape
    logger.info(f"  Resampled volume shape: {resampled_shape}")
    
    nii_file = os.path.join(config['SEGMENTATIONS_DIR'], f"{series_id}.nii")
    
    if not os.path.exists(nii_file):
        logger.error(f"  ❌ ERROR: Mask file not found: {nii_file}")
        return
    
    try:
        gt_mask_resampled = process_nii_mask(nii_file, original_shape, resampled_shape)
        
        liver_mask_resampled = (gt_mask_resampled == 1).astype(np.uint8)
        liver_voxels = liver_mask_resampled.sum()
        
        logger.info(f"  Liver mask voxels (before cleaning): {liver_voxels:,}")
        
        if liver_voxels == 0:
            logger.error(f"  ❌ ERROR: No liver voxels in mask!")
            return
        
        labeled_mask, num_components = nd_label(liver_mask_resampled)
        
        if num_components > 1:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component = component_sizes.argmax()
            liver_mask_resampled = (labeled_mask == largest_component).astype(np.uint8)
            logger.info(f"  Removed {num_components - 1} small artifacts, kept {liver_mask_resampled.sum():,} voxels")
        
        liver_coords = np.where(liver_mask_resampled > 0)
        
        bbox_z_min_orig = liver_coords[0].min()
        bbox_z_max_orig = liver_coords[0].max()
        bbox_y_min_orig = liver_coords[1].min()
        bbox_y_max_orig = liver_coords[1].max()
        bbox_x_min_orig = liver_coords[2].min()
        bbox_x_max_orig = liver_coords[2].max()
        
        bbox_center_orig = np.array([
            (bbox_z_min_orig + bbox_z_max_orig) / 2,
            (bbox_y_min_orig + bbox_y_max_orig) / 2,
            (bbox_x_min_orig + bbox_x_max_orig) / 2
        ])
        
        bbox_size_orig = np.array([
            bbox_z_max_orig - bbox_z_min_orig,
            bbox_y_max_orig - bbox_y_min_orig,
            bbox_x_max_orig - bbox_x_min_orig
        ])
        
        logger.info(f"  Original bbox size: {bbox_size_orig}")
        
        volume_final, offsets = center_crop_or_pad(volume_resampled, config['INPUT_DIMENSIONS'])
        liver_mask_final, _ = center_crop_or_pad(liver_mask_resampled, config['INPUT_DIMENSIONS'])
        
        bbox_center_final = bbox_center_orig - offsets
        bbox_size_final = bbox_size_orig
        
        logger.info(f"  Final bbox center: {bbox_center_final}")
        logger.info(f"  Final bbox size: {bbox_size_final}")
        logger.info(f"  Liver mask voxels (final): {liver_mask_final.sum():,}")
        
        coverage = bbox_size_final / np.array(config['INPUT_DIMENSIONS']) * 100
        logger.info(f"  BBox coverage: Z={coverage[0]:.1f}%, Y={coverage[1]:.1f}%, X={coverage[2]:.1f}%")
        
        np.savez_compressed(
            output_file,
            volume=volume_final.astype(np.float32),
            bbox_center=bbox_center_final.astype(np.float32),
            bbox_size=bbox_size_final.astype(np.float32),
            mask=liver_mask_final.astype(np.uint8)
        )
        
        logger.info(f"✅ Saved: {series_id}_liver.npz")
    
    except Exception as e:
        logger.error(f"❌ Failed processing {series_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


def process_series_wrapper(series_tuple, config, injury_df):
    """Wrapper for multiprocessing."""
    series_path, series_id = series_tuple
    try:
        process_series(series_path, series_id, config, injury_df)
    except Exception as e:
        logger.error(f"✗ Failed {series_id}: {e}")


def discover_labeled_series(config):
    """Discover series with segmentation masks."""
    segmentation_files = glob.glob(os.path.join(config['SEGMENTATIONS_DIR'], "*.nii"))
    labeled_series_ids = {os.path.splitext(os.path.basename(f))[0] for f in segmentation_files}
    
    labeled_series = []
    
    for patient_dir in glob.glob(os.path.join(config['TRAIN_IMAGES_DIR'], "*")):
        for series_path in glob.glob(os.path.join(patient_dir, "*")):
            series_id = os.path.basename(series_path)
            
            if series_id in labeled_series_ids:
                labeled_series.append((series_path, series_id))
    
    return labeled_series


def run_pipeline():
    """Process LIVER LOCALIZATION: Full abdomen + tight liver bbox."""
    
    logger.info("LIVER LOCALIZATION PREPROCESSING (FINAL VERSION)")
    
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
    
    injury_df = pd.read_csv(CONFIG['LABELS_CSV'])
    injury_df['has_liver'] = ((injury_df['liver_low'] == 1) | 
                               (injury_df['liver_high'] == 1)).astype(int)
    
    liver_injured_count = injury_df['has_liver'].sum()
    logger.info(f"Series with LIVER injury: {liver_injured_count}")
    
    labeled_series = discover_labeled_series(CONFIG)
    
    logger.info(f"Found {len(labeled_series)} labeled series")
    logger.info(f"Expected liver injury series: ~{liver_injured_count}")
    
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
    logger.info(f"Using {num_workers} parallel workers")
    
    process_func = partial(process_series_wrapper, config=CONFIG, injury_df=injury_df)
    
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        pool.map(process_func, labeled_series)
    
    elapsed_time = time.time() - start_time
    
    liver_files = glob.glob(os.path.join(CONFIG['OUTPUT_DIR'], "*_liver.npz"))
    
    logger.info("PREPROCESSING COMPLETED!")
    logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
    logger.info(f"Successfully saved: {len(liver_files)} liver files")
    logger.info(f"Output directory: {CONFIG['OUTPUT_DIR']}")


if __name__ == '__main__':
    run_pipeline()
