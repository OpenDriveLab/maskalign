# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from transformers import CLIPVisionModel, ViTModel
import pdb


def resize_pos_embed(x):
    # [256, C] -> [196, C]
    C = x.shape[-1]
    x = x.reshape(1, 16, 16, C).permute(0, 3, 1, 2)
    x = F.interpolate(x, (14, 14), mode='bicubic', align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(196, C)
    return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 loss_weights="mean", mask_type="random", fusion_type="simple", target_norm="none", loss_type="l2",
                 head_type="linear", teacher_model="openai/clip-vit-base-patch16"):
        super().__init__()

        assert loss_weights in ["mean", "out", "linear_decay"] or "top" in loss_weights or "mid" in loss_weights
        self.loss_weights = loss_weights
        assert mask_type in ["random", "attention"]
        self.mask_type = mask_type
        assert fusion_type in ["simple", "linear", "sum"]
        self.fusion_type = fusion_type
        assert target_norm in ["none", "l2", "whiten", "bn"]
        self.target_norm = target_norm
        assert loss_type in ["l2", "l1", "smoothl1"]
        self.loss_type = loss_type
        assert head_type in ["linear", "norm_linear", "mlp", "mlp2"]
        self.head_type= head_type
        # assert "clip" in teacher_model or "dino" in teacher_model
        self.teacher_model_name = teacher_model

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        if "clip-vit-base-patch16" in self.teacher_model_name or "dino-vitb16" in self.teacher_model_name:
            target_dim = 768
            teacher_depth = 12
        else:
            target_dim = 1024
            teacher_depth = 24

        if self.head_type == "linear":
            self.distill_heads = nn.ModuleList([nn.Linear(embed_dim, target_dim) for i in range(teacher_depth)])
        elif self.head_type == "norm_linear":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                                                    norm_layer(embed_dim),
                                                    nn.Linear(embed_dim, target_dim)
                                                )
                                    for i in range(teacher_depth)])
        elif self.head_type == "mlp":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                                                    nn.Linear(embed_dim, embed_dim),
                                                    nn.GELU(),
                                                    nn.Linear(embed_dim, target_dim)
                                                )
                                    for i in range(teacher_depth)])
        elif self.head_type == "mlp2":
            self.distill_heads = nn.ModuleList([nn.Sequential(
                                                    nn.Linear(embed_dim, embed_dim),
                                                    norm_layer(embed_dim),
                                                    nn.Linear(embed_dim, target_dim)
                                                )
                                    for i in range(teacher_depth)])

        if self.fusion_type == "linear":
            # only len(student) == len(teacher)
            self.distill_weights = nn.Parameter(torch.eye(len(self.blocks)) + 0.01, requires_grad=True)
        elif self.fusion_type == "sum":
            self.distill_weights = nn.Parameter(torch.ones(teacher_depth, len(self.blocks)) / len(self.blocks), requires_grad=True)

        self.initialize_weights()

        if "clip" in self.teacher_model_name:
            self.clip_model = CLIPVisionModel.from_pretrained(self.teacher_model_name)
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False
                if "clip-vit-large-patch14" in self.teacher_model_name and "position_embedding" in name:
                    param.data = torch.cat([param.data[:1], resize_pos_embed(param.data[1:])], dim=0)
            if "clip-vit-large-patch14" in self.teacher_model_name:
                self.clip_model.vision_model.embeddings.position_ids = torch.arange(197).expand((1, -1))

        elif "dino" in self.teacher_model_name:
            self.dino_model = ViTModel.from_pretrained(self.teacher_model_name)
            for param in self.dino_model.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def denormalize(self, images, type="imagenet"):
        # sr_images [B, 3, H, W]
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1).type_as(images)
        return std*images + mean

    def normalize(self, images, type="clip"):
        # images [B, 3, h, w]
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).type_as(images)
        return (images - mean) / std

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, ids_keep
    
    def attention_masking(self, x, mask_ratio, importance):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = importance.to(x.device) # large is keep, small is remove
        
        # sort noise for each sample
        ids_shuffle = torch.multinomial(noise, L, replacement=False)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_dump = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, ids_keep

    def forward_encoder(self, x, mask_ratio, attentions):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask_type == "attention":
            importance = attentions[-1][:, :, 0, 1:].mean(1)
            x, ids_keep = self.attention_masking(x, mask_ratio, importance)
        else:
            x, ids_keep = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        hidden_states = []
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.norm(x)

        return hidden_states, ids_keep

    @torch.no_grad()
    def forward_clip(self, x):
        if "clip-vit-large-patch14" in self.teacher_model_name:
            x = F.interpolate(x, (196, 196), mode='bicubic', align_corners=False)
            
        x = self.normalize(self.denormalize(x))
        input = {
            "pixel_values": x,
            "output_hidden_states": True,
            "output_attentions": True
        }
        outputs = self.clip_model(**input)
        
        last_hidden_state, pooler_output, hidden_states, attentions = outputs[0], outputs[1], outputs[2], outputs[3]
        return last_hidden_state, pooler_output, hidden_states, attentions
    
    @torch.no_grad()
    def forward_dino(self, x):
        input = {
            "pixel_values": x,
            "output_hidden_states": True,
            "output_attentions": True
        }
        outputs = self.dino_model(**input)
        
        last_hidden_state, pooler_output, hidden_states, attentions = outputs[0], outputs[1], outputs[2], outputs[3]
        return last_hidden_state, pooler_output, hidden_states, attentions
    

    def get_student(self, hidden_states):
        student = hidden_states
        if self.fusion_type != "simple":
            student = [x.unsqueeze(0) for x in student]
            student = torch.cat(student, dim=0)
            student = torch.einsum('ab,bcde->acde', self.distill_weights, student)
            student = torch.chunk(student, student.shape[0], dim=0)
            student = [x.squeeze(0) for x in student]
        student = [self.distill_heads[i](x) for i, x in enumerate(student)]
        return student

    def get_teacher(self, hidden_states, ids_keep):
        teacher = []
        for i in range(1, len(hidden_states)):
            y = hidden_states[i]
            if self.target_norm == "l2":
                y = F.normalize(y, dim=-1)
            elif self.target_norm == "whiten":
                y = F.layer_norm(y, (y.shape[-1],))
            elif self.target_norm == "bn":
                y = (y - y.mean()) / (y.var() + 1.e-6)**.5
            cls = y[:, :1, :]
            y = y[:, 1:, :]
            y = torch.gather(y, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
            teacher.append(torch.cat([cls, y], dim=1))
        return teacher

    def forward_loss(self, student, teacher):
        """
        student: ([B*4, L//4, C]...)
        teacher: ([B, 1+L, C]...)
        ids_shuffle: [B, L] 
        """
        loss = torch.tensor(0., device=student[0].device)
        
        if self.loss_weights == "mean":
            weight_list = [1/len(student)]*len(student)
        elif self.loss_weights == "out":
            weight_list = [0.]*(len(student)-1) + [1.]
        elif self.loss_weights == "linear_decay":
            weight_list_ = list(range(len(student)))
            weight_list = [i / sum(weight_list_) for i in weight_list_]
        elif "top" in self.loss_weights:  # topk
            topk = int(self.loss_weights[3:])
            weight_list = [0.] * (len(student)-topk) + [1/topk] * topk
        elif "mid" in self.loss_weights:
            mid = int(self.loss_weights[3:])
            weight_list = [0.] * mid + [1.] + [0.] * (len(student) - mid - 1)

        for i, x in enumerate(student):
            y = teacher[i]
            if weight_list[i] > 0:
                if self.loss_type == "l2":
                    loss = loss + weight_list[i] * ((y - x) ** 2).mean()
                elif self.loss_type == "smoothl1":
                    loss = loss + weight_list[i] * 2 * F.smooth_l1_loss(y, x)
                elif self.loss_type == "l1":
                    loss = loss + weight_list[i] * F.l1_loss(y, x)
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        if "clip" in self.teacher_model_name:
            _, _, hidden_states_teacher, attentions = self.forward_clip(imgs)
        elif "dino" in self.teacher_model_name:
            _, _, hidden_states_teacher, attentions = self.forward_dino(imgs)
        hidden_states, ids_keep = self.forward_encoder(imgs, mask_ratio, attentions)
        student = self.get_student(hidden_states)
        teacher = self.get_teacher(hidden_states_teacher, ids_keep)
        loss = self.forward_loss(student, teacher)
        return loss


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16  
mae_vit_large_patch16 = mae_vit_large_patch16  
  