#!/bin/bash

# Process 3 smaller AVIRIS-NG datasets (4-5m pixel size, manageable file sizes)

source venv/bin/activate

DATASETS=(
    "/raid/AVIRIS_NG/imagery/ang20190624t214359_rdn_v2u1/ang20190624t214359_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190624t230039_rdn_v2u1/ang20190624t230039_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190624t230448_rdn_v2u1/ang20190624t230448_rdn_v2u1_img"
)

OUTPUT_BASE="outputs/dataset_small"
PATCH_SIZE=256
STRIDE=128

echo "========================================================================"
echo "Processing ${#DATASETS[@]} AVIRIS-NG datasets (smaller files)"
echo "Output: $OUTPUT_BASE"
echo "Patch size: ${PATCH_SIZE}x${PATCH_SIZE}, stride: $STRIDE"
echo "========================================================================"

for ds in "${DATASETS[@]}"; do
    filename=$(basename "$ds" "_img")
    output_dir="${OUTPUT_BASE}/${filename}"
    
    echo ""
    echo "========================================"
    echo "Processing: $filename"
    echo "Output: $output_dir"
    echo "========================================"
    
    python scripts/test_data_pipeline.py \
        --aviris-file "$ds" \
        --output-dir "$output_dir" \
        --patch-size $PATCH_SIZE \
        2>&1 | tee "${output_dir}.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $filename"
    else
        echo "✗ ERROR processing $filename"
    fi
done

echo ""
echo "========================================================================"
echo "Dataset generation complete!"
echo ""
echo "Summary:"
find "$OUTPUT_BASE" -name "*.h5" | while read f; do
    patches=$(h5dump -H "$f" 2>/dev/null | grep "DATASPACE" | head -1 | grep -oP '\(\K[0-9]+' || echo "?")
    echo "  - $(basename $(dirname $f)): $patches patches"
done
echo ""
echo "Thumbnails:"
find "$OUTPUT_BASE" -type d -name "*_thumbnails" | while read d; do
    count=$(ls -1 "$d"/*.png 2>/dev/null | wc -l)
    echo "  - $(basename $d): $count thumbnails"
done
echo "========================================================================"
