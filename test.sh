export CUDA_VISIBLE_DEVICES=0,2,3
export CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.run --nproc_per_node=3 Retrieval_inv.py \
--config /data1/wjy/speed-up/kaiyuan/configs/Retrieval_flickr.yaml \
--output_dir /data1/wjy/speed-up/ALBEF-speedup/ALBEF_97/output/f30k \
--text_encoder /data1/wjy/speed-up/ALBEF-speedup/ALBEF_97/configs/bert-base-uncased \
--checkpoint /data1/wjy/speed-up/ALBEF-speedup/ALBEF_Classifier/ckpt/flickr30k.pth \
--evaluate \
--ann \
