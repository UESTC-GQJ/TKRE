SEED=42
python train.py \
    --data_dir ../../datasets/example_data \
    --train_filename merged_train.json \
    --data_test_dir ../../datasets/ \
    --model_name_or_path /home/guoquanjiang/DSARE/hf-models/AI-ModelScope/roberta-large \
    --input_format typed_entity_marker_punct \
    --seed $SEED \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --project_name TACRED \
    --run_name run-1