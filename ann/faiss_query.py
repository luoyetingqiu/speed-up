import json

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from utils import Time

default_para_dict = {
    'd': 512,
    'M': 16,
    'nbits': 8,
    'nlists': 128,
    'quantizer': faiss.IndexFlatIP(512),
    'metric': faiss.METRIC_INNER_PRODUCT,
}

model_dict = {
    'IndexFlatL2': ['d'],
    'IndexFlatIP': ['d'],
    'IndexHNSWFlat': ['d', 'M'],
    'IndexIVFFlat': ['quantizer', 'd', 'nlists', 'metric'],
    'IndexLSH': ['d', 'nbits'],
    'IndexScalarQuantizer': ['d'],
    'IndexPQ': ['d', 'M', 'nbits'],
    'IndexIVFPQ': ['quantizer', 'd', 'nlists', 'M', 'nbits'],
}


def faiss_query_onebyone_time_eval(query_feat: np.ndarray, gallery_feat: np.ndarray, index_name: str,
                gallery_ids: list = None, topk=128, train_feat=None, **para_dict):
    default_para_dict.update(para_dict)

    assert index_name in model_dict.keys(), 'index name is not valid, check spell please'
    assert query_feat.shape[1] == gallery_feat.shape[1], "query and gallery features' size mismatch"

    if train_feat is None:
        train_feat = gallery_feat

    default_para_dict['d'] = query_feat.shape[1]
    default_para_dict['quantizer'] = eval('faiss.IndexFlatIP')(default_para_dict['d'])

    query_feat, gallery_feat = torch.Tensor(query_feat), torch.Tensor(gallery_feat)
    query_feat = F.normalize(query_feat, p=2, dim=1)
    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
    query_feat, gallery_feat = query_feat.numpy(), gallery_feat.numpy()
    paras = [default_para_dict[key] for key in model_dict[index_name]]
    index = eval('faiss.' + index_name)(*paras)

    index.train(train_feat)
    assert index.is_trained
    index.add(gallery_feat)

    time_log = Time()
    time_log.update_last_time()
    D, top_res = index.search(query_feat, topk)
    query_time = time_log.cost()

    return query_time


def faiss_query_time_eval(query_feat: np.ndarray, gallery_feat: np.ndarray, index_name: str,
                gallery_ids: list = None, topk=128, train_feat=None, **para_dict):
    default_para_dict.update(para_dict)

    assert index_name in model_dict.keys(), 'index name is not valid, check spell please'
    assert query_feat.shape[1] == gallery_feat.shape[1], "query and gallery features' size mismatch"

    if train_feat is None:
        train_feat = gallery_feat

    default_para_dict['d'] = query_feat.shape[1]
    default_para_dict['quantizer'] = eval('faiss.IndexFlatIP')(default_para_dict['d'])

    query_feat, gallery_feat = torch.Tensor(query_feat), torch.Tensor(gallery_feat)
    query_feat = F.normalize(query_feat, p=2, dim=1)
    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
    query_feat, gallery_feat = query_feat.numpy(), gallery_feat.numpy()
    paras = [default_para_dict[key] for key in model_dict[index_name]]
    index = eval('faiss.' + index_name)(*paras)

    index.train(train_feat)
    assert index.is_trained
    index.add(gallery_feat)

    time_log = Time()
    time_log.update_last_time()
    for one_query in query_feat:
        D, top_res = index.search(one_query[np.newaxis, :], topk)
    query_fps = query_feat.shape[0] / time_log.cost()

    return query_fps


def faiss_query(query_feat: np.ndarray, gallery_feat: np.ndarray, index_name: str,
                gallery_ids: list = None, topk=128, train_feat=None, **para_dict):
    default_para_dict.update(para_dict)

    assert index_name in model_dict.keys(), 'index name is not valid, check spell please'
    assert query_feat.shape[1] == gallery_feat.shape[1], "query and gallery features' size mismatch"

    if train_feat is None:
        train_feat = gallery_feat

    default_para_dict['d'] = query_feat.shape[1]
    default_para_dict['quantizer'] = eval('faiss.IndexFlatIP')(default_para_dict['d'])

    query_feat, gallery_feat = torch.Tensor(query_feat), torch.Tensor(gallery_feat)
    query_feat = F.normalize(query_feat, p=2, dim=1)
    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
    query_feat, gallery_feat = query_feat.numpy(), gallery_feat.numpy()
    paras = [default_para_dict[key] for key in model_dict[index_name]]
    index = eval('faiss.' + index_name)(*paras)

    index.train(train_feat)
    assert index.is_trained
    index.add(gallery_feat)
    D, top_res = index.search(query_feat, topk)
    top_res = [(top_res[i][top_res[i] >= 0]).tolist() for i in range(top_res.shape[0])]
    if gallery_ids:
        top_res = [[gallery_ids[i] for i in top] for top in top_res]
    return top_res


if __name__ == "__main__":
    image_embeds = torch.load('/data/ALBEF/embeds/flickr_test_image_embeds.pt', map_location='cpu').numpy()
    text_embeds = torch.load('/data/ALBEF/embeds/flickr_test_text_embeds.pt', map_location='cpu').numpy()
    ann_alg = 'IndexFlatIP'

    img_time = faiss_query_onebyone_time_eval(image_embeds[np.newaxis, 0], text_embeds, ann_alg, topk=128)
    text_time = faiss_query_onebyone_time_eval(text_embeds[np.newaxis, 0], image_embeds, ann_alg, topk=128)

    print(f"{ann_alg} Img-2-Text (μs): {img_time * 10**6:.1f}")
    print(f"{ann_alg} Text-2-Img (μs): {text_time * 10**6:.1f}")
