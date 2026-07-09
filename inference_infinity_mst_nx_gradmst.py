from inference_treeformer import relation_infer as _relation_infer


def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False):
    return _relation_infer(
        h,
        out,
        net,
        obj_token,
        rln_token,
        nms=nms,
        map_=map_,
        mst=True,
    )
