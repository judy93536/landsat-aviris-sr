#!/bin/bash

# Process all 5 good AVIRIS-NG datasets with 4-5m pixel size

source venv/bin/activate

DATASETS=(
    "/raid/AVIRIS_NG/imagery/ang20190623t194727_rdn_v2u1/ang20190623t194727_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190624t214359_rdn_v2u1/ang20190624t214359_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190624t230039_rdn_v2u1/ang20190624t230039_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190624t230448_rdn_v2u1/ang20190624t230448_rdn_v2u1_img"
    "/raid/AVIRIS_NG/imagery/ang20190715t172845_rdn_v2v2/ang20190715t172845_rdn_v2v2_img"
)

OUTPUT_BASE="outputs/full_dataset"
PATCH_SIZE=256
STRIDE=128

echo "========================================================================"
echo "Processing ${#DATASETS[@]} AVIRIS-NG datasets"
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
        2>&1 | tee "$output_dir.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $filename"
    else
        echo "✗ ERROR processing $filename"
    fi
done

echo ""
echo "========================================================================"
echo "Dataset generation complete!"
echo "Summary:"
find "$OUTPUT_BASE" -name "*.h5" -exec echo "  - {}" \;
echo "========================================================================"
