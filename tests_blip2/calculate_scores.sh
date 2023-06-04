#!/bin/bash

# This script calculates the scores for the test using the precalculated captions and the ground truth captions.

# Calculate the scores for each caption file
for caption_file in "output/captions/"*; do

    # Get the name of the file without the extension
    name=$(basename "$caption_file")
    name="${name%.*}"

    # Skip the references file
    if [[ $name == "references" ]]; then
        continue
    fi

    echo "Calculating scores for ${name}"

    python clipscore/clipscore.py "${caption_file}" output/images/ --references output/captions/references.json --save_per_instance "scores_instance_${name}.json"

    # Move the scores file to the output folder
    mkdir -p "output/scores/${name}"
    mv scores* "output/scores/${name}"

done
