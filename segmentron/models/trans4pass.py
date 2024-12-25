import torch
import torch.nn.functional as F

from segmentron.config import cfg
from segmentron.models.model_zoo import MODEL_REGISTRY
from segmentron.models.segbase import SegBaseModel
# # --- dmlpv1
# from segmentron.modules.dmlp import DMLP
# --- dmlpv2
from segmentron.modules.dmlpv2 import DMLP

__all__ = ['Trans4PASSOneShareBackboneMoE']


@MODEL_REGISTRY.register(name='Trans4PASSOneShareBackboneMoE')
class Trans4PASSOneShareBackboneMoE(SegBaseModel):

    def __init__(self, num_source_domains=2):
        super().__init__()
        vit_params = cfg.MODEL.TRANS2Seg
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 32) ** 2
        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['nclass'] = self.nclass
        vit_params['emb_chans'] = cfg.MODEL.EMB_CHANNELS
        self.num_source_domains = num_source_domains
        self.dede_heads = torch.nn.ModuleList([DMLP(vit_params) for _ in range(self.num_source_domains)])

        hidden_dim = 128
        self.moe_conv = torch.nn.ModuleList([
            torch.nn.Conv2d(vit_params['emb_chans'] * num_source_domains, hidden_dim, 3, 1, 1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim, num_source_domains, 3, 1, 1),
        ])

        self.pred = torch.nn.Conv2d(vit_params['emb_chans'], vit_params['nclass'], 1)

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        feats_all = []
        x_all = []
        for i in range(self.num_source_domains):
            feats, x = self.dede_heads[i](c1, c2, c3, c4)
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            feats = sum(feats)
            feats_all.append(feats)
            x_all.append(x)

        x_stacked = torch.stack(feats_all, dim=1)
        x = torch.cat(feats_all, dim=1)
        for conv_layer in self.moe_conv:
            x = conv_layer(x)
        moe_weights = torch.softmax(x, dim=1)
        feats_moe = (x_stacked * moe_weights.unsqueeze(2)).sum(1)
        x = self.pred(feats_moe)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x_all, x
