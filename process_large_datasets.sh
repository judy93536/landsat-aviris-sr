#!/bin/bash
# Process large AVIRIS datasets to generate training patches
# These are the 2 large datasets that will generate 200+ additional patches

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Output directory
OUTPUT_BASE="outputs/dataset_large"
mkdir -p "$OUTPUT_BASE"

# Large datasets (will take 2-3 hours each)
DATASETS=(
    "/raid/AVIRIS_NG/imagery/ang20190623t194727_rdn_v2u1/ang20190623t194727_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190715t172845_rdn_v2v2/ang20190715t172845_rdn_v2v2_img"
)

PATCH_SIZE=256
STRIDE=128

echo "========================================================================"
echo "Processing Large AVIRIS Datasets"
echo "========================================================================"
echo "Start time: $(date)"
echo "Output directory: $OUTPUT_BASE"
echo "Patch size: ${PATCH_SIZE}x${PATCH_SIZE}"
echo "Stride: $STRIDE"
echo ""
echo "Datasets to process: ${#DATASETS[@]}"
for dataset in "${DATASETS[@]}"; do
    echo "  - $(basename $dataset)"
done
echo "========================================================================"
echo ""

# Process each dataset
for i in "${!DATASETS[@]}"; do
    AVIRIS_FILE="${DATASETS[$i]}"
    DATASET_NAME=$(basename "$AVIRIS_FILE" _img)
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET_NAME"

    echo ""
    echo "------------------------------------------------------------------------"
    echo "[$((i+1))/${#DATASETS[@]}] Processing: $DATASET_NAME"
    echo "------------------------------------------------------------------------"
    echo "AVIRIS file: $AVIRIS_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "Started: $(date)"
    echo ""

    # Run pipeline
    python scripts/test_data_pipeline.py \
        --aviris-file "$AVIRIS_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --patch-size $PATCH_SIZE \
        2>&1 | tee "$OUTPUT_DIR.log"

    echo ""
    echo "Completed: $(date)"

    # Show summary
    if [ -f "$OUTPUT_DIR/test_patches.h5" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/test_patches.h5" | cut -f1)
        echo "Generated HDF5 file: $SIZE"
    fi

    if [ -d "$OUTPUT_DIR/test_patches_thumbnails" ]; then
        NUM_THUMBS=$(ls -1 "$OUTPUT_DIR/test_patches_thumbnails" | wc -l)
        echo "Generated thumbnails: $NUM_THUMBS"
    fi
    echo ""
done

echo ""
echo "========================================================================"
echo "Large Dataset Generation Complete"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Summary:"
for dataset in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename "$dataset" _img)
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET_NAME"

    if [ -f "$OUTPUT_DIR/test_patches.h5" ]; then
        SIZE=$(du -h "$OUTPUT_DIR/test_patches.h5" | cut -f1)
        NUM_PATCHES=$(python -c "import h5py; f=h5py.File('$OUTPUT_DIR/test_patches.h5','r'); print(f.attrs['n_patches'])" 2>/dev/null || echo "?")
        echo "  $DATASET_NAME: $NUM_PATCHES patches ($SIZE)"
    else
        echo "  $DATASET_NAME: FAILED"
    fi
done
echo "========================================================================"
