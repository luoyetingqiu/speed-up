This repo is the official implementation of "[Efficient Image-Text Retrieval via Keyword-Guided Pre-Screening](https://arxiv.org/abs/2303.07740)" in PyTorch.

### Requirements:
* python 3.8
* opencv-python 4.7.0.72
* ruamel-yaml 0.17.21 
* pytorch 1.8.0
* transformers 4.8.1
* timm 0.4.9

### Download
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth"> ALBEF checkpoint for retrieval on MSCOCO </a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth"> ALBEF checkpoint for retrieval on Flickr30k </a>
* Download <a href="https://cocodataset.org/#download"> MSCOCO </a> or <a href="http://shannon.cs.illinois.edu/DenotationGraph/"> Flickr30k </a>      datasets from the original websites
* Download and extract <a href="https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"> MSCOCO </a> or <a      href="https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json"> Flickr30k </a> dataset json files
* Download <a href="https://drive.google.com/drive/folders/1hQmytQZN61chxW9VYRFM2RNUqzNuxUmu?usp=drive_link"> Screen </a> json files
* Download <a href="https://huggingface.co/bert-base-uncased"> bert-base-uncased </a> from huggingface to the configs folder

### Experiment:
In configs/Retrieval_coco.yaml or configs/Retrieval_flickr.yaml, set the paths for the json files and the image path 
1. Test time and ann_time using 4 3090 GPUs  (time: set method="baseline" in config, ann time: set method="ann" in config):
<pre>python -m torch.distributed.run --nproc_per_node=4 Retrieval_inv.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/flickr30k \
--text_encode ./configs/bert-base-uncased \
--checkpoint [Pretrained checkpoint] \
--evaluate \
--time_eval</pre> 

2. Test ++Ours using 4 3090 GPUs  (set screen file paths in config):
<pre>python -m torch.distributed.run --nproc_per_node=4 Retrieval_inv.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/flickr30k \
--text_encode ./configs/bert-base-uncased \
--checkpoint [Pretrained checkpoint] \
--evaluate \
--screen </pre> 

3. Test ann_retrieval using 4 3090 GPUs  (set ann in config):
<pre>python -m torch.distributed.run --nproc_per_node=4 Retrieval_inv.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/flickr30k \
--text_encode ./configs/bert-base-uncased \
--checkpoint [Pretrained checkpoint] \
--evaluate \
--ann </pre> 

4. Test ALBEF_all and +ALBEF_0 using 4 3090 GPUs  (ALBEF_all: set k_test="all" in config, +ALBEF_0: set k_test=128 in config):
<pre>python -m torch.distributed.run --nproc_per_node=4 Retrieval_inv.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir ./output/flickr30k \
--text_encode ./configs/bert-base-uncased \
--checkpoint [Pretrained checkpoint] \
--evaluate</pre> 

### Acknowledgement
Our code is extended from the following repositories. We thank the authors for releasing the codes.
* [ALBEF](https://github.com/salesforce/ALBEF)
