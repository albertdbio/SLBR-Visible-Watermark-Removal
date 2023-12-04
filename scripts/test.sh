CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --dataset_dir /content/SLBR-Visible-Watermark-Removal/Watermark/CLWD \
  --nets slbr \
  --models slbr \
  --input_width 256 \
  --input_height 256 \
  --test-batch 1 \
  --evaluate \
  --preprocess none \
  --no_flip true \
  --name slbr_v1 \
  --mask_mode res \
  --k_center 2 \
  --dataset CLWD \
  --use_refine \
  --k_refine 3 \
  --k_skip_stage 3 \
  # --resume /content/SLBR-Visible-Watermark-Removal/Watermark/slbr_v1/model_best.pth.tar \
