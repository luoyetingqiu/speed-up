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


### Experiment:

1. Download MSCOCO or Flickr30k datasets from the original websites.
2. Download and extract <a href="https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json"> MSCOCO </a> or <a href="https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json"> Flickr30 </a> dataset json files.
3. Download <a href="https://huggingface.co/bert-base-uncased"> bert-base-uncased </a> from huggingface to the configs folder
3. In configs/Retrieval_coco.yaml or configs/Retrieval_flickr.yaml, set the paths for the json files and the image path.
4. 
5. Test time using 4 3090 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=3 Retrieval_inv.py \
--config ./configs/Retrieval_flickr_inv_97.yaml \
--output_dir ./output/flickr30k \
--text_encode ./configs/bert-base-uncased \
--checkpoint [Pretrained checkpoint] \
--evaluate \
--ann \
--time_eval</pre> 

### Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes.

* [ALBEF](https://github.com/salesforce/ALBEF)
