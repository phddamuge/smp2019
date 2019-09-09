export DATA_DIR=./

python3 ./run_classifier.py \
  --task_name=dialogue \
  --do_train=False\
  --do_eval=False \
  --do_predict=True \
  --data_dir=$DATA_DIR \
  --vocab_file=./vocab.txt \
  --bert_config_file=./bert_config.json \
  --init_checkpoint=./model.ckpt-5158\
  --max_seq_length=50 \
  --train_batch_size=5 \
  --eval_batch_size=5 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --output_dir=./
