SEED=42
python /data/guoquanjiang/DSARE/codes/traditional_re/few_train.py \
    --data_dir /data/guoquanjiang/DSARE/datasets/example_data \
    --train_filename few_train.json \
    --data_test_dir /data/guoquanjiang/DSARE/datasets/ \
    --model_name_or_path /data/guoquanjiang/DSARE/codes/traditional_re/output/relation_mlm \
    --input_format typed_entity_marker_punct \
    --seed $SEED \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 50 \
    --project_name TACRED \
    --run_name run-1