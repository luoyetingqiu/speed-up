import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from toolz import partition_all

import utils
from ann.faiss_query import faiss_query, faiss_query_time_eval, faiss_query_onebyone_time_eval
from dataset import create_dataset, create_single_loader
from models.model_retrieval import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
from utils import Screen


@torch.no_grad()
def evaluation_baseline_time(model, time_data_loader, tokenizer, device, config):
    model.eval()

    print("============================ Baseline Evaluation ==============================")
    time_log = utils.Time()
    texts = time_data_loader.dataset.text

    num_text = len(texts)
    text_bs = 1
    text_feats = []
    text_embeds = []
    text_atts = []

    time_log.update_last_time()
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        # text_input.input_ids: 256 * 30, text_input.attention_mask: 256 * 30
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        # text_feat: 256 * 30 * 768
        text_feat = text_output.last_hidden_state
        # text_embed: 256 * 256
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
        break
    print(f"Baseline Text Processing: {time_log.cost() * 1000}")

    img_num = len(time_data_loader.dataset)
    image_feats = []
    image_embeds = []
    time_log.update_last_time()
    for image, img_order in time_data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
        break
    print(f"Baseline Image Processing: {time_log.cost() * 1000}")

    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    encoder_output = image_feats[0].unsqueeze(0)
    encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

    time_log.update_last_time()
    output = model.text_encoder(encoder_embeds=text_feats[0].unsqueeze(0),
                                attention_mask=text_atts[0].unsqueeze(0),
                                encoder_hidden_states=encoder_output,
                                encoder_attention_mask=encoder_att,
                                return_dict=True,
                                mode='fusion'
                                )
    score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
    fusion_time = time_log.cost()
    print(f"Baseline Text Fusion:{fusion_time * img_num * 1000}")
    print(f"Baseline Image Fusion:{fusion_time * num_text * 1000}")


@torch.no_grad()
def evaluation_ann_time(model, time_data_loader, tokenizer, device, config):
    model.eval()

    print("============================ Ann Evaluation ==============================")
    texts = time_data_loader.dataset.text

    text_eval_bs = 256
    text_num = len(texts)
    text_embeds = []

    for i in range(0, text_num, text_eval_bs):
        text = texts[i: min(text_num, i + text_eval_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.cpu())

    img_num = len(time_data_loader.dataset)
    image_embeds = []
    for image, img_order in time_data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed.cpu())

    text_embeds = torch.cat(text_embeds, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    ann_param = config['ann']
    ann_alg = config['ann']['alg']

    image_embeds = image_embeds.cpu().numpy()
    text_embeds = text_embeds.cpu().numpy()

    img_time = faiss_query_onebyone_time_eval(image_embeds[np.newaxis, 0], text_embeds, ann_alg, **ann_param)
    text_time = faiss_query_onebyone_time_eval(text_embeds[np.newaxis, 0], image_embeds, ann_alg, **ann_param)

    print(f"{ann_alg} Img (μs): {img_time}")
    print(f"{ann_alg} Text (μs): {text_time}")






@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, img2text_screen, text2img_screen):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    time_log = utils.Time()

    dataset = data_loader.dataset
    texts = dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs), desc='text feature extraction'):
        # list: 256
        text = texts[i: min(num_text, i + text_bs)]
        # text_input.input_ids: 256 * 30, text_input.attention_mask: 256 * 30
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(
            device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        # text_feat: 256 * 30 * 768
        text_feat = text_output.last_hidden_state
        # text_embed: 256 * 256
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed.cpu())
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())

    print('Text feature extraction time: {} ms'.format(time_log.cost()))

    # val: text_embeds: 5070 * 256
    text_embeds = torch.cat(text_embeds, dim=0)
    # val: text_feats: 5070 * 30 * 256
    text_feats = torch.cat(text_feats, dim=0)
    # val: text_atts: 5070 * 30
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_order in tqdm(data_loader, desc='image feature extraction'):
        # image: 64 * 3 * 384 * 384
        image = image.to(device)
        # image_feat: 64 * 577 * 768
        image_feat = model.visual_encoder(image)
        # image_embed: 64 * 256
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed.cpu())

    print('Image feature extraction time: {} ms'.format(time_log.cost()))

    # val: image_feats: 1014 * 577 * 768
    image_feats = torch.cat(image_feats, dim=0)
    # val: image_embeds: 1014 * 256
    image_embeds = torch.cat(image_embeds, dim=0)

    # val: sims_matrix: 1014 * 5070
    sims_matrix = image_embeds @ text_embeds.t()

    

    if args.save and utils.is_main_process():
        torch.save(image_embeds.cpu(), config['img_feat_save_file'])
        torch.save(text_embeds.cpu(), config['text_feat_save_file'])

    # val: score_matrix_i2t: 1014 * 5070
    score_matrix_i2t = torch.full((len(dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    if args.ann:
        ann_param = config['ann']
        ann_alg = config['ann']['alg']
        if args.screen:
            ...
        else:
            ann_i2t = faiss_query(image_embeds.cpu().numpy(), text_embeds.cpu().numpy(), ann_alg, **ann_param)
            ann_t2i = faiss_query(text_embeds.cpu().numpy(), image_embeds.cpu().numpy(), ann_alg, **ann_param)
    # ann_i2t = json.load(open(config['ann_i2t']))
    # ann_t2i = json.load(open(config['ann_t2i']))

    time_log.update_last_time()
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        sims = sims.to(device)
        if args.ann:
            topk_idx = ann_i2t[start + i]
        else:
            if args.screen:
                topk_idx = img2text_screen[start + i]
                if len(topk_idx) > config['k_test']:
                    topk_sim, topk_idx_in_inv = sims[topk_idx].topk(k=config['k_test'], dim=0)
                    topk_idx = [topk_idx[j] for j in topk_idx_in_inv]
            else:
                if config['k_test'] == 'all':
                    topk_idx = range(sims_matrix.size(1))
                else:
                    topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)


        for cap_order in partition_all(config['batch_size_cal'], topk_idx):
            cap_order = list(cap_order)
            # encoder_output: 128 * 577 * 768
            encoder_output = image_feats[start + i].repeat(len(cap_order), 1, 1).to(device)
            # encoder_att: 128 * 577
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            # text_feats[topk_idx]: 128 * 30 * 768, text_atts[topk_idx]: 128 * 30,
            # output.last_hidden_state: 128 * 30 * 768
            output = model.text_encoder(encoder_embeds=text_feats[cap_order].to(device),
                                        attention_mask=text_atts[cap_order].to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
            # score: list 128
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, cap_order] = score
    print('TR fusion time: {} ms'.format(time_log.cost()))

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, len(dataset.image)), -100.0).to(device)  # score_matrix_t2i: 5070 * 1014

    step = sims_matrix.size(0) // num_tasks
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        sims = sims.to(device)
        if args.ann:
            topk_idx = ann_t2i[start + i]
        else:
            if args.screen:
                topk_idx = text2img_screen[start + i]
                if len(topk_idx) > config['k_test']:
                    topk_sim, topk_idx_in_inv = sims[topk_idx].topk(k=config['k_test'], dim=0)
                    topk_idx = [topk_idx[j] for j in topk_idx_in_inv]
            else:
                if config['k_test'] == 'all':
                    topk_idx = range(sims_matrix.size(1))
                else:
                    topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        for img_order in partition_all(config['batch_size_cal'], topk_idx):
            img_order = list(img_order)
            encoder_output = image_feats[img_order].to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(encoder_embeds=text_feats[start + i].repeat(len(img_order), 1, 1).to(device),
                                        attention_mask=text_atts[start + i].repeat(len(img_order), 1).to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
            score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, img_order] = score

    print('IR fusion time: {} ms'.format(time_log.cost()))

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    print('Total evaluation time {} ms'.format(time_log.total_cost()))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])  # 60val: ranks: 60
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print('Arguments:')
    print(args)

    print('Config:')
    print(config)

    #### Dataset ####
    print("Creating retrieval dataset")
    # number of images -->> train_dataset: 145000, val_dataset: 1014, test_dataset: 1000
    test_dataset = create_dataset('re_split', config)

    test_loader = create_single_loader(test_dataset, None,
                                       batch_size=config['batch_size_test'],
                                       num_workers=4,
                                       is_train=False,
                                       collate_fn=None)

    img2text_screen, text2img_screen = None, None
    if args.screen:
        img2text_id_screen = json.load(open(config['img2text_file'], 'r'))
        text2img_id_screen = json.load(open(config['text2img_file'], 'r'))
        img2text_screen, text2img_screen = Screen.screen_id2order(img2text_id_screen, text2img_id_screen,
                                                                  test_dataset.img_id2order, test_dataset.text_id2order)
        img2text_recall, text2img_recall = Screen.recall(img2text_screen, text2img_screen,
                                                         test_dataset.img2txt, test_dataset.txt2img)
        avg_i2t_screen, avg_t2i_screen = Screen.avg_len(img2text_screen, text2img_screen)
        del img2text_id_screen
        del text2img_id_screen

        print(f'i2t recall: {img2text_recall}')
        print(f't2i recall: {text2img_recall}')
        print(f'i2t avg screen length: {int(avg_i2t_screen)}')
        print(f't2i avg screen length: {int(avg_t2i_screen)}')

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    # utils.count_param(model)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if args.evaluate:
            state_dict = checkpoint
        else:
            state_dict = checkpoint['model']
            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        # print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    print("Start evaluating!")

    if args.time_eval:
        if config['method'] == 'baseline':
            evaluation_baseline_time(model_without_ddp, test_loader, tokenizer, device, config)
        elif config['method'] == 'ann':
            evaluation_ann_time(model_without_ddp, test_loader, tokenizer, device, config)
        exit(0)

    start_time = time.time()
    score_result_i2t, score_result_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config, img2text_screen, text2img_screen)

    if utils.is_main_process():
        test_result = itm_eval(score_result_i2t, score_result_t2i, test_dataset.txt2img, test_dataset.img2txt)
        print(test_result)

        log_stats = {
            **{f'test_{k}': v for k, v in test_result.items()}
        }
        with open(os.path.join(args.output_dir, "log_inv.txt"), "a") as f:
            f.write("checkpoint: " + args.checkpoint + "\n")
            f.write("config: " + args.config + "\n")
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='/data1/wjy/speed-up/ALBEF-speedup/ALBEF_97/configs/bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ann', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--time_eval', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
