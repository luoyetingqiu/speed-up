import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import faiss

para_dict = {
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
    'IndexIVFPQ': ['quantizer','d','nlists','M','nbits'],
}


def faiss_query(query_feat: np.ndarray, gallery_feat: np.ndarray, index_name: str,
                gallery_ids: list=[], topk=128, train_feat=None):
    assert index_name in model_dict.keys(), 'index name is not valid, check spell please'
    assert query_feat.shape[1] == gallery_feat.shape[1], "query and gallery features' size mismatch"

    if train_feat is None:
        train_feat = gallery_feat

    para_dict['d'] = query_feat.shape[1]
    para_dict['quantizer'] = eval('faiss.IndexFlatIP')(para_dict['d'])

    query_feat, gallery_feat = torch.Tensor(query_feat), torch.Tensor(gallery_feat)
    query_feat = F.normalize(query_feat, p=2, dim=1)
    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
    query_feat, gallery_feat = query_feat.numpy(), gallery_feat.numpy()
    paras = [para_dict[key] for key in model_dict[index_name]]
    index = eval('faiss.'+index_name)(*paras)

    index.train(train_feat)
    assert index.is_trained
    index.add(gallery_feat)
    start_time = time.time()
    D, top_res = index.search(query_feat, topk)
    match_time = time.time()-start_time
    print(match_time)
    # top_res = [(top_res[i][top_res[i] >= 0]).tolist() for i in range(top_res.shape[0])]
    # if gallery_ids:
    #     top_res = [[gallery_ids[i] for i in top] for top in top_res]
    # return top_res, match_time
    return None, match_time


def main():
    np.random.seed(42)

    d = 256
    query_num, gallery_num = 10000, 771837

    query, gallery = np.random.randn(query_num, d), np.random.randn(gallery_num, d)
    query, gallery = query.astype(np.float32), gallery.astype(np.float32)

    tm_lst = []
    try_times = 29
    assert try_times % 2
    for i in np.random.choice(query_num, try_times):
        res, tm = faiss_query(query[np.newaxis, i], gallery, 'IndexFlatIP')
        # res, tm = faiss_query(query[np.newaxis, i], gallery, 'IndexPQ')
        # res, tm = faiss_query(query[np.newaxis, i], gallery, 'IndexIVFPQ')
        tm_lst.append(tm)

    print(f'middle match time(ms): {sorted(tm_lst)[try_times//2] * 10 ** 3:.1f}')
    # print('corresponding fps:', 1/sorted(tm_lst)[try_times//2])


if __name__ == '__main__':
    main()