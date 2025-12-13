# For training with Finerva
# step 1
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --data_root data-finerva/price \
  --model models/lightweight_models/MBZUAI_LaMini-Flan-T5-77M \
  --use_fin price \
  --user_msg rationale \
  --img_type vit \
  --bs 2 \
  --eval_bs 4 \
  --epoch 20 \
  --lr 5e-5 \
  --output_len 512 \
  --use_caption \
  --use_generate \
  --prompt_format QCM-E \
  --output_dir experiments-1210-price-train

# step 2
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --data_root data-finerva/pact \
  --model models/lightweight_models/MBZUAI_LaMini-Flan-T5-783M \
  --use_fin pact \
  --user_msg answer \
  --img_type vit \
  --bs 2 \
  --eval_bs 4 \
  --epoch 20 \
  --lr 5e-5 \
  --output_len 512 \
  --use_caption \
  --use_generate \
  --prompt_format QCMG-EA \
  --output_dir experiments-0806-pact-train \
  --eval_le experiments-0806-pact-train/pact_rationale_models-lightweight_models-MBZUAI_LaMini-Flan-T5-783M_vit_QCM-E_lr5e-05_bs4_op512_ep20/predictions_ans_eval.json \
  --test_le experiments-0806-pact-train/pact_rationale_models-lightweight_models-MBZUAI_LaMini-Flan-T5-783M_vit_QCM-E_lr5e-05_bs4_op512_ep20/predictions_ans_test.json