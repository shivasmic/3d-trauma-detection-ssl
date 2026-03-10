"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

""" This script creates holdout samples using the same preprocessing protocol for n number of samples which are exclusive 
to the already created 1206 samples, this is extremely useful in generating more volumes for UNet evaluation or
generating more volumes for Semi-Supervised training for VDETR
"""

import os
import glob
import pydicom
import numpy as np
import scipy.ndimage
import nibabel as nib
from dotenv import load_dotenv
from log import get_preprocessing_logger
from skimage.measure import regionprops, label
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
    "OUTPUT_DIR": os.getenv("HOLDOUT_OUTPUT_DIR"),  
    "EXISTING_DATA_DIR": os.getenv("OUTPUT_DIR"),  
    "NUM_HOLDOUT_SAMPLES": 250,  
}

logger = get_preprocessing_logger()

def load_dicom_series(series_path):
    dicom_files = glob.glob(os.path.join(series_path, "*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {series_path}")
    
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(slices[0].SliceThickness)
    except AttributeError:
        pixel_spacing = [1.0, 1.0]
        slice_thickness = 1.0
    
    current_spacing = np.array([slice_thickness] + list(pixel_spacing), dtype=np.float32)
    image = np.stack([s.pixel_array for s in slices], axis=0).astype(np.int16)
    
    rescale_slope = slices[0].get('RescaleSlope', 1)
    rescale_intercept = slices[0].get('RescaleIntercept', 0)
    image = image * rescale_slope + rescale_intercept
    
    return image, current_spacing


def resample_and_normalize(image, current_spacing, config):
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
    nii_mask = nib.load(nii_path)
    mask_data = np.round(nii_mask.get_fdata()).astype(np.uint8)
    
    if mask_data.shape != series_original_shape:
        if mask_data.shape == (series_original_shape[2], series_original_shape[1], series_original_shape[0]):
            mask_data = np.transpose(mask_data, (2, 1, 0))
        elif mask_data.shape == (series_original_shape[1], series_original_shape[2], series_original_shape[0]):
            mask_data = np.transpose(mask_data, (2, 0, 1))
        else:
            logger.warning(f"    WARNING: Mask shape {mask_data.shape} doesn't match DICOM shape {series_original_shape}")
    
    resize_factor = np.array(series_resampled_shape) / np.array(mask_data.shape)
    mask_aligned_float = scipy.ndimage.zoom(mask_data, resize_factor, order=0)
    mask_aligned = (mask_aligned_float > 0.5).astype(np.uint8)
    final_z, final_y, final_x = series_resampled_shape
    mask_aligned = mask_aligned[:final_z, :final_y, :final_x]
    
    return mask_aligned


def size_standardize(volume, final_shape, is_mask=False):
    final_z, final_y, final_x = final_shape
    
    if is_mask:
        volume = volume.astype(np.int32)
    else:
        volume = volume.astype(np.float32)
    
    padding_value = 0
    pad_z, pad_y, pad_x = [max(0, fs - ds) for fs, ds in zip(final_shape, volume.shape)]
    
    padding = [(pad_z // 2, pad_z - pad_z // 2),
               (pad_y // 2, pad_y - pad_y // 2),
               (pad_x // 2, pad_x - pad_x // 2)]
    
    volume_padded = np.pad(volume, padding, mode='constant', constant_values=padding_value)
    
    start_z = (volume_padded.shape[0] - final_z) // 2
    start_y = (volume_padded.shape[1] - final_y) // 2
    start_x = (volume_padded.shape[2] - final_x) // 2
    
    final_standardized_volume = volume_padded[
        start_z : start_z + final_z,
        start_y : start_y + final_y,
        start_x : start_x + final_x
    ]
    
    if final_standardized_volume.shape != final_shape:
        raise ValueError(f"Shape error: Final volume shape {final_standardized_volume.shape} != target shape {final_shape}")
    
    if is_mask:
        return final_standardized_volume.astype(np.uint8)
    
    return final_standardized_volume


def size_standardize_with_crop_indices(volume, final_shape, start_indices, is_mask=False):
    final_z, final_y, final_x = final_shape
    
    if is_mask:
        volume = volume.astype(np.int32)
    else:
        volume = volume.astype(np.float32)
    
    padding_value = 0
    pad_z, pad_y, pad_x = [max(0, fs - ds) for fs, ds in zip(final_shape, volume.shape)]
    
    if any([pad_z > 0, pad_y > 0, pad_x > 0]):
        padding = [(pad_z // 2, pad_z - pad_z // 2),
                   (pad_y // 2, pad_y - pad_y // 2),
                   (pad_x // 2, pad_x - pad_x // 2)]
        volume = np.pad(volume, padding, mode='constant', constant_values=padding_value)
    
    if start_indices is None:
        start_z = (volume.shape[0] - final_z) // 2
        start_y = (volume.shape[1] - final_y) // 2
        start_x = (volume.shape[2] - final_x) // 2
    else:
        start_z, start_y, start_x = start_indices
        start_z = max(0, min(start_z, volume.shape[0] - final_z))
        start_y = max(0, min(start_y, volume.shape[1] - final_y))
        start_x = max(0, min(start_x, volume.shape[2] - final_x))
    
    final_volume = volume[
        start_z : start_z + final_z,
        start_y : start_y + final_y,
        start_x : start_x + final_x
    ]
    
    if final_volume.shape != final_shape:
        raise ValueError(f"Shape error: {final_volume.shape} != {final_shape}")
    
    if is_mask:
        return final_volume.astype(np.uint8)
    
    return final_volume


def process_series(series_path, series_id, config):
    try:
        raw_image, current_spacing = load_dicom_series(series_path)
    except Exception as e:
        logger.error(f"Skipping series {series_id}: DICOM loading error: {e}")
        return
    
    original_shape = raw_image.shape
    
    temp_resampled_volume = resample_and_normalize(raw_image, current_spacing, config)
    resampled_shape = temp_resampled_volume.shape    
    nii_file = os.path.join(config['SEGMENTATIONS_DIR'], f"{series_id}.nii")
    
    if os.path.exists(nii_file):
        try:
            nii_mask = nib.load(nii_file)
            mask_data_raw = np.round(nii_mask.get_fdata()).astype(np.uint8)
            
            gt_mask_temp = process_nii_mask(nii_file, original_shape, resampled_shape)
            
            labeled_temp = label(gt_mask_temp > 0)
            props_temp = regionprops(labeled_temp)
            
            if props_temp:
                largest_temp = max(props_temp, key=lambda p: p.area)
                min_z_full, min_y_full, min_x_full, max_z_full, max_y_full, max_x_full = largest_temp.bbox
                
                injury_size = np.array([
                    max_z_full - min_z_full,
                    max_y_full - min_y_full,
                    max_x_full - min_x_full
                ])
                logger.info(f"  Injury size (Z, Y, X): {injury_size}")
                
                center_z = (min_z_full + max_z_full) // 2
                center_y = (min_y_full + max_y_full) // 2
                center_x = (min_x_full + max_x_full) // 2
                
                final_z, final_y, final_x = config['INPUT_DIMENSIONS']
                start_z = center_z - final_z // 2
                start_y = center_y - final_y // 2
                start_x = center_x - final_x // 2
                
                crop_indices = (start_z, start_y, start_x)
                logger.info(f"  Injury center: z={center_z}, y={center_y}, x={center_x}")
                logger.info(f"  Crop indices: z={start_z}, y={start_y}, x={start_x}")
                
                if injury_size[0] > config['INPUT_DIMENSIONS'][0]:
                    logger.warning(f"Injury Z ({injury_size[0]}) exceeds crop size ({config['INPUT_DIMENSIONS'][0]})")
                if injury_size[1] > config['INPUT_DIMENSIONS'][1]:
                    logger.warning(f"Injury Y ({injury_size[1]}) exceeds crop size ({config['INPUT_DIMENSIONS'][1]})")
                if injury_size[2] > config['INPUT_DIMENSIONS'][2]:
                    logger.warning(f"Injury X ({injury_size[2]}) exceeds crop size ({config['INPUT_DIMENSIONS'][2]})")
            else:
                crop_indices = None
                logger.info(f"No injury found, using center crop")
            
            final_volume = size_standardize_with_crop_indices(
                temp_resampled_volume, config['INPUT_DIMENSIONS'], crop_indices, is_mask=False
            )
            gt_mask = size_standardize_with_crop_indices(
                gt_mask_temp, config['INPUT_DIMENSIONS'], crop_indices, is_mask=True
            )
            
            
            labeled_mask = label(gt_mask.astype(bool))
            props_all = regionprops(labeled_mask)
            
            
            gt_bbox = None
            
            if props_all:
                largest_prop = max(props_all, key=lambda prop: prop.area)
                gt_mask_clean = (labeled_mask == largest_prop.label).astype(np.uint8)
                
                min_z, min_y, min_x, max_z, max_y, max_x = largest_prop.bbox
                logger.info(f"Raw bbox coords: z[{min_z}:{max_z}], y[{min_y}:{max_y}], x[{min_x}:{max_x}]")
                
                gt_center = np.array([
                    (min_z + max_z) / 2,
                    (min_y + max_y) / 2,
                    (min_x + max_x) / 2
                ])
                gt_size = np.array([
                    max_z - min_z,
                    max_y - min_y,
                    max_x - min_x
                ])
                
                gt_bbox = {'center': gt_center, 'size': gt_size}
                final_gt_mask = gt_mask_clean
            else:
                final_gt_mask = np.zeros(config['INPUT_DIMENSIONS'], dtype=np.uint8)
            
            if gt_bbox:
                np.savez_compressed(
                    os.path.join(config['OUTPUT_DIR'], f"{series_id}_labeled.npz"),
                    volume=final_volume.astype(np.float32),
                    bbox_center=gt_bbox['center'].astype(np.float32),
                    bbox_size=gt_bbox['size'].astype(np.float32),
                    mask=final_gt_mask.astype(np.uint8)
                )
                logger.info(f"Saved Labeled Data: {series_id}.npz (Mask Sum: {final_gt_mask.sum()})")
                logger.info(f"BBox Center (Z, Y, X): {gt_bbox['center']}")
                logger.info(f"BBox Size (Z, Y, X): {gt_bbox['size']}")
                logger.info(f"Volume shape: {final_volume.shape}")
        
        except Exception as e:
            logger.error(f"Skipping mask processing for {series_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    else:
        final_volume = size_standardize(temp_resampled_volume, config['INPUT_DIMENSIONS'])
        np.savez_compressed(
            os.path.join(config['OUTPUT_DIR'], f"{series_id}_unlabeled.npz"),
            volume=final_volume.astype(np.float32)
        )
        logger.info(f"Saved Unlabeled Data: {series_id}.npz (No .nii file found)")


def process_series_wrapper(series_tuple, config):
    series_path, series_id, has_label = series_tuple
    try:
        start_time = time.time()
        process_series(series_path, series_id, config)
        elapsed = time.time() - start_time
        label_status = "labeled" if has_label else "unlabeled"
    except Exception as e:
        logger.error(f"Failed {series_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())


def get_already_processed_series(existing_data_dir):
    if not os.path.exists(existing_data_dir):
        logger.warning(f"Existing data directory not found: {existing_data_dir}")
        return set()
    
    processed_files = glob.glob(os.path.join(existing_data_dir, "*.npz"))
    processed_ids = set()
    
    for f in processed_files:
        basename = os.path.basename(f)
        if basename.endswith('_labeled.npz'):
            series_id = basename[:-len('_labeled.npz')]
        elif basename.endswith('_unlabeled.npz'):
            series_id = basename[:-len('_unlabeled.npz')]
        else:
            series_id = basename[:-4] 
        
        processed_ids.add(series_id)
    
    if len(processed_ids) > 0:
        sample_ids = sorted(list(processed_ids))[:10]
        logger.info(f"Sample already processed IDs: {sample_ids}...")
    
    return processed_ids


def discover_new_series(config):
    already_processed = get_already_processed_series(config['EXISTING_DATA_DIR'])
    
    segmentation_files = glob.glob(os.path.join(config['SEGMENTATIONS_DIR'], "*.nii"))
    labeled_series_ids = {os.path.splitext(os.path.basename(f))[0] for f in segmentation_files}
    
    logger.info(f"Total segmentation masks available: {len(labeled_series_ids)}")
    
    new_labeled_series = []
    new_unlabeled_series = []
    
    for patient_dir in glob.glob(os.path.join(config['TRAIN_IMAGES_DIR'], "*")):
        for series_path in glob.glob(os.path.join(patient_dir, "*")):
            series_id = os.path.basename(series_path)
            
            if series_id in already_processed:
                continue
            
            # Add to appropriate list
            if series_id in labeled_series_ids:
                new_labeled_series.append((series_path, series_id, True))
            else:
                new_unlabeled_series.append((series_path, series_id, False))
    
    logger.info(f"NEW series discovered (not yet processed):")
    logger.info(f"NEW Labeled: {len(new_labeled_series)}")
    logger.info(f"NEW Unlabeled: {len(new_unlabeled_series)}")
    logger.info(f"TOTAL NEW: {len(new_labeled_series) + len(new_unlabeled_series)}")
    
    return new_labeled_series, new_unlabeled_series


def run_pipeline():
    logger.info("HOLDOUT SET PREPROCESSING PIPELINE")
    logger.info(f"Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        os.makedirs(CONFIG['OUTPUT_DIR'])
        logger.info(f"Created output directory: {CONFIG['OUTPUT_DIR']}")
    
    new_labeled_series, new_unlabeled_series = discover_new_series(CONFIG)
    
    all_new_series = new_labeled_series + new_unlabeled_series
    
    if len(all_new_series) == 0:
        logger.error("ERROR: No new series found to process!")
        logger.error("All series have already been processed.")
        return
    
    num_to_process = min(CONFIG['NUM_HOLDOUT_SAMPLES'], len(all_new_series))
    series_to_process = all_new_series[:num_to_process]
    
    logger.info("PROCESSING PLAN:")
    logger.info(f"Available NEW series: {len(all_new_series)}")
    logger.info(f"Target holdout samples: {CONFIG['NUM_HOLDOUT_SAMPLES']}")
    logger.info(f"Will process: {num_to_process} series")
    logger.info(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
    logger.info(f"Using {num_workers} parallel workers")
    
    process_func = partial(process_series_wrapper, config=CONFIG)    
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        pool.map(process_func, series_to_process)
    
    elapsed_time = time.time() - start_time
    
    logger.info("HOLDOUT PREPROCESSING COMPLETED!")
    logger.info(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f}s)")
    logger.info(f"Average time per series: {elapsed_time/len(series_to_process):.2f}s")
    logger.info(f"Processed {len(series_to_process)} NEW series")
    logger.info(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    
    labeled_processed = sum(1 for _, _, is_labeled in series_to_process if is_labeled)
    unlabeled_processed = len(series_to_process) - labeled_processed
    logger.info("SUMMARY:")
    logger.info(f"  - Labeled holdout: {labeled_processed}")
    logger.info(f"  - Unlabeled holdout: {unlabeled_processed}")
    logger.info(f"  - Total holdout: {len(series_to_process)}")


if __name__ == '__main__':
    run_pipeline()
