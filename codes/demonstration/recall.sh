SEED=42
python recall_demonstration.py \
    --data_dir ../../datasets/example_data \
    --train_filename train.json \
    --data_test_dir ../../datasets/example_data \
    --model_name_or_path ../../hf-models/roberta-large \
    --input_format typed_entity_marker_punct \
    --seed $SEED \
    --test_batch_size 4 \
    --checkpoint_filepath ../traditional_re/wandb/run-xxxx/files/checkpoints/epoch_xxxx

# checkpoint_filepath refers to the selected checkpoints filepath