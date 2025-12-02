prefix="../../datasets/llm_corpus_context" 
file_end=".json"
python corpus.py \
    --demo_path /data/guoquanjiang/DSARE/datasets/example_data/few_train.json \
    --auto_modelpath /data/guoquanjiang/Llama-2-13b-chat-hf \
    --output_dir $prefix$file_end \
    --dataset tacred \
    --k 6;
# dataset: tacred, tacrev, retacred

# python merge_data.py