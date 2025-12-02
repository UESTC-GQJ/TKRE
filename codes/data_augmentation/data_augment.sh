prefix="../../datasets/zephyr_generated" 
file_end=".json"
python automodel_da.py \
    --demo_path /data/guoquanjiang/DSARE/datasets/example_data/few_train.json \
    --auto_modelpath ../../hf-models/zephyr-7b-alpha \
    --output_dir $prefix$file_end \
    --dataset tacred \
    --k 8;
# dataset: tacred, tacrev, retacred

# python merge_data.py