K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res

INPUT_SIZE=256
DATASET=CLWD
NAME=slbr_v1

CUDA_VISIBLE_DEVICES=0 python3  test.py \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --dataset_dir /content/SLBR-Visible-Watermark-Removal/Watermark/${DATASET} \
  --no_flip true \
  --name ${NAME} \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --dataset ${DATASET} \
  --resume /content/SLBR-Visible-Watermark-Removal/Watermark/${NAME}/model_best.pth.tar \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP}
  
    # --checkpoint /media/sda/Watermark \
  
