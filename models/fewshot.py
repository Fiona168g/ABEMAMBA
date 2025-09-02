import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron

from .Transformer import Transformer
from .PSPNet import OneModel as PSPNet


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask 
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 
    tmp_supp = s          
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 
    similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
    similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity

class Discriminator(nn.Module): 
    def __init__(self, indim, outdim=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(indim, indim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(indim//2, indim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(indim//4, outdim),
            nn.Sigmoid(),  # add by wy  
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 100  # number of foreground partitions
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)
        self.layer_norm = nn.LayerNorm(512)
        fea_dim = 1024 + 512       
        reduce_dim = 64 
        self.transformer = None   

        self.part = nn.ModuleList([MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),
                                   MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),
                                   MyCrossAttention(reduce_dim, part_num=14, num_heads=4, attn_drop=0.1, proj_drop=0.1),])


        #PSPNet###############################
        PSPNet_ = PSPNet()
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        ############################
        
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),              
            nn.Dropout2d(p=0.5)                 
        )
     
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Dropout2d(p=0.5)               
        )
    
        self.query_merge = nn.Sequential(
            nn.Conv2d(512+2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.binary_cls = nn.ModuleList([Discriminator(reduce_dim, 1), Discriminator(reduce_dim, 1), Discriminator(reduce_dim, 1)])
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        if self.transformer is None:
            self.transformer = Transformer(shot=self.n_shots).to(self.device)
        self.n_queries = len(qry_imgs)
        self.iter = 3
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        align_loss = torch.zeros(1, device=self.device)
        mse_loss = torch.zeros(1, device=self.device)
        qry_loss = torch.zeros(1, device=self.device)

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        supp_imgs_t = torch.stack(
            [torch.stack(way, dim=0) for way in supp_imgs], dim=0
        ).permute(2, 0, 1, 3, 4, 5).contiguous()

        outputs = []
        loss_D_total, loss_G_total = None, None

        for epi in range(supp_bs):
            q = qry_imgs[0][epi:epi+1]
            h, w = q.shape[-2:]
            _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(q)
            query_feat = torch.cat([query_feat_3, query_feat_2], dim=1)
            query_feat = self.down_query(query_feat)
            qry_fts_epi = query_feat

            s_x = torch.stack([supp_imgs_t[epi, 0, sh] for sh in range(self.n_shots)], dim=0)
            s_y = supp_mask[epi, 0, :self.n_shots, ...]
            mask = (s_y == 1).float().unsqueeze(1)

            supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], dim=1)
            supp_feat = self.down_supp(supp_feat)

            supp_feat_bin_each = Weighted_GAP(
                supp_feat,
                F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                            mode="bilinear", align_corners=True)
            )
            supp_feat_bin = supp_feat_bin_each.mean(0, keepdim=True)

            supp_prototype = supp_feat_bin.clone().squeeze(-1).squeeze(-1)
            query_feat_flatten = query_feat.flatten(-2)
            prototype_norm = F.normalize(supp_prototype, dim=1)
            query_feat_flatten_norm = F.normalize(query_feat_flatten, dim=1)
            cosine_sim = torch.einsum("bc,bcl->bl", prototype_norm, query_feat_flatten_norm) / 0.1
            intial_prediction = cosine_sim.reshape(-1, 1, supp_feat_3.size(2), supp_feat_3.size(3))
            intial_prediction[intial_prediction < 0.7] = 0

            s_y_like = s_y.unsqueeze(0)
            similarity2 = get_similarity(query_feat_4, supp_feat_4, s_y_like)
            similarity1 = get_similarity(query_feat_5, supp_feat_5, s_y_like)
            similarity = torch.cat([similarity1, similarity2], dim=1)

            supp_feat_bin_spatial = supp_feat_bin.repeat(1, 1, supp_feat.shape[-2], supp_feat.shape[-1])
            supp_fts_epi = supp_feat.mean(0, keepdim=True)

            supp_feat_merged = self.supp_merge(torch.cat([supp_fts_epi, supp_feat_bin_spatial], dim=1))
            query_feat_cond = self.query_merge(torch.cat([query_feat, supp_feat_bin, similarity * 10], dim=1))
            multi_query_feat = self.transformer(query_feat_cond, similarity)

            multi_refined_query = []
            hw_shapes = [a.shape[-2:] for a in multi_query_feat]
            fg_map = intial_prediction.float()
            out_refined, weights_rf, fused_query_feat = self.transformer.query_cross(
                multi_query_feat, fg_map, similarity, hw_shapes
            )
            multi_refined_query.append(fused_query_feat)

            meta_out_soft = out_refined.softmax(1)
            final_out = torch.cat([meta_out_soft[:, 0:1], meta_out_soft[:, 1:]], dim=1)

            _fg = final_out[:, 1]
            _fg[final_out[:, 0] > final_out[:, 1]] = 0
            _fg = (_fg.unsqueeze(1) > 0.5).float()
            out_refined_r1, _, fused_query_feat_r1 = self.transformer.refine1(fused_query_feat, _fg, similarity)
            multi_refined_query.append(fused_query_feat_r1)

            meta_out_soft = out_refined_r1.softmax(1)
            final_out = torch.cat([meta_out_soft[:, 0:1], meta_out_soft[:, 1:]], dim=1)

            _fg = final_out[:, 1]
            _fg[final_out[:, 0] > final_out[:, 1]] = 0
            _fg = (_fg.unsqueeze(1) > 0.5).float()
            out_refined_r2, _, fused_query_feat_r2 = self.transformer.refine2(fused_query_feat_r1, _fg, similarity)
            multi_refined_query.append(fused_query_feat_r2)

            meta_out_soft = out_refined_r2.softmax(1)
            final_out = torch.cat([meta_out_soft[:, 0:1], meta_out_soft[:, 1:]], dim=1)

            fg_map = final_out[:, 1].unsqueeze(1)
            gt_map = (qry_mask[epi] == 1).float().unsqueeze(0).unsqueeze(1)
            gt_map = F.interpolate(gt_map, size=fg_map.shape[-2:], mode="bilinear", align_corners=True)

            loss_D, loss_G = None, None
            for ly, fused_feat in enumerate(multi_refined_query):
                bz_, c_, h_, w_ = fused_feat.shape
                source = fused_feat.flatten(2).permute(0, 2, 1).contiguous()
                gt_map_ = F.interpolate(gt_map, size=fused_feat.shape[-2:], mode="bilinear", align_corners=True)
                gt_map_ = gt_map_.flatten(2).permute(0, 2, 1).contiguous()
                fg_map_ = F.interpolate(fg_map, size=fused_feat.shape[-2:], mode="bilinear", align_corners=True)
                fg_map_ = fg_map_.flatten(2).permute(0, 2, 1).contiguous()

                source_r = source * gt_map_
                source_f = source * fg_map_
                part_real, _ = self.part[ly](source_r, gt_map_)
                part_fake, _ = self.part[ly](source_f, fg_map_)

                A = self.cos(part_fake, part_real)
                index = torch.min(A, dim=1)[1]
                out_fake = torch.stack([part_fake[i, index[i], :] for i in range(0, part_fake.size(0))], dim=0)
                out_real = torch.stack([part_real[i, index[i], :] for i in range(0, part_real.size(0))], dim=0)

                out_fake = self.binary_cls[ly](out_fake)
                out_real = self.binary_cls[ly](out_real)

                device = out_fake.device
                pseudo_T = torch.ones((out_fake.size(0), 1), device=device)
                pseudo_T1 = 0.9 * torch.ones((out_fake.size(0), 1), device=device)
                pseudo_F = torch.zeros(out_fake.size(0), 1, device=device)

                loss_G = (loss_G + self.binary_loss(out_fake, pseudo_T).mean()) if (loss_G is not None) \
                        else self.binary_loss(out_fake, pseudo_T).mean()
                loss_d2 = self.binary_loss(out_real, pseudo_T1)
                loss_d1 = self.binary_loss(out_fake, pseudo_F)
                loss_D = (loss_D + 0.5 * (loss_d1.mean() + loss_d2.mean())) if (loss_D is not None) \
                        else 0.5 * (loss_d1.mean() + loss_d2.mean())

            loss_D_total = (loss_D_total + loss_D) if (loss_D_total is not None) else loss_D
            loss_G_total = (loss_G_total + loss_G) if (loss_G_total is not None) else loss_G

            final_out_up = F.interpolate(final_out, size=(h, w), mode="bilinear", align_corners=True)
            preds = final_out_up

            if train:
                align_loss_epi = self.alignLoss(supp_fts_epi, qry_fts_epi, preds, supp_mask[epi])
                align_loss = align_loss + align_loss_epi

                fg_prototypes = [supp_feat_bin.squeeze(0)]
                proto_mse_loss_epi = self.proto_mse(qry_fts_epi, preds, supp_mask[epi], fg_prototypes)
                mse_loss = mse_loss + proto_mse_loss_epi

                qry_fts_ = [[self.getFeatures(qry_fts_epi, qry_mask[epi])]]
                qry_prototypes = self.getPrototype(qry_fts_)
                thr = self.thresh_pred[0] if isinstance(self.thresh_pred, (list, tuple)) else self.thresh_pred
                qry_pred_tmp = self.getPred(qry_fts_epi, qry_prototypes[0], thr)
                if qry_pred_tmp.dim() == 3:
                    qry_pred_tmp = qry_pred_tmp.unsqueeze(1)
                qry_pred_tmp = F.interpolate(qry_pred_tmp, size=img_size, mode="bilinear", align_corners=True)
                preds_tmp = torch.cat((1.0 - qry_pred_tmp, qry_pred_tmp), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds_tmp, eps, 1 - eps))
                qry_loss = qry_loss + self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

            outputs.append(final_out_up)

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, align_loss, mse_loss, qry_loss, loss_D_total, loss_G_total

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes


    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)


                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def proto_mse(self, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]

                fg_prototypes = self.getPrototype(qry_fts_)

                fg_prototypes_ = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0)
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes, dim=0), dim=0)

                # Combine prototypes from different scales
                # fg_prototypes = self.alpha * fg_prototypes[way]
                # fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                # supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                # supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss

                loss_sim += self.criterion_MSE(fg_prototypes_, supp_prototypes_)

        return loss_sim
class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

class MyCrossAttention(nn.Module):
    def __init__(self, dim, part_num=2, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        self.dropout = 0.1

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)#
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.ass_drop   = nn.Dropout(0.1)
        
        self.parts = nn.Parameter(torch.rand(part_num, dim))

        self.drop_prob = 0.1
        self.layer_norms = nn.LayerNorm(dim)
        self.ffn = FFN(dim, 3*dim, dropout=self.dropout)


    def forward(self, supp_feat, supp_mask=None):
        # q： query_feat: B,L,C
        # k=v: masked support feat: B,L,C
        # supp_mask: B,L,C  # 0,1
        # supp_mask = 1 - supp_mask
        k = supp_feat #
        v = k.clone()
        B, _, _ = k.shape
        q_ori = self.parts.unsqueeze(0).repeat(B,1,1) #[B,N,C]
        _, N, C = q_ori.shape
        N_s = k.size(1)

        q = self.q_fc(q_ori).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #
    
        k = self.k_fc(k).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) #[b,n,L,c]
        
        # if supp_mask is not None:
        #     supp_mask = supp_mask.permute(0,2,1).contiguous().repeat(1, self.num_heads, 1) # [bs, nH, n]

        
        '''using the cosine similarity instead of dot product'''
        # shape: [B,n_h,L,c]
        # attn = (q @ k.transpose(-2, -1)) * self.scale # [bs, nH, nq, ns]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.einsum("bhqc,bhsc->bhqs", q, k) / 0.1 #0.1 is temprature

        # if supp_mask is not None:
        #     supp_mask = supp_mask.unsqueeze(-2).float() # [bs, nH, 1, ns]
        #     supp_mask = supp_mask * -10000.0
        #     attn = attn + supp_mask       

        
        attn_out = attn.mean(1) #[B,N,HW]
        attn_out = F.sigmoid(attn_out)
        attn_out = attn_out * (supp_mask.permute(0,2,1)) #B,N,HW * B  将注意力均值与支持掩码相乘
        
        attn = attn.softmax(dim=-1) #[b,N,ns]
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x =  x + q_ori
        x = self.ffn(x)
        x = self.layer_norms(x)
        # print(attn_out.shape)
        return x, attn_out # [b,L,C]






