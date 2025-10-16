#!/bin/sh

cd sam

python scripts/amg.py --checkpoint sam_vit_b_01ec64.pth --model-type vit_b --input input/ --output output/
