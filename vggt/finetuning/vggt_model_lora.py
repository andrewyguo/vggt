import argparse
import json
import os 

import torch 
import loralib as lora 
from pathlib import Path

from vggt.models.vggt import VGGT

class LoRAVGGT(VGGT):
    def __init__(self, **kwargs):
        self.args = None
        super().__init__(**kwargs)

    def setup_lora_model(self, args, mark_lora=True):
        self.args = args

        self.replace_aggregator_patch_embed()
        self.replace_aggregator_frame_blocks()
        self.replace_aggregator_global_blocks()
        self.replace_camera_head()

        if mark_lora:
            lora.mark_only_lora_as_trainable(self)
    
    def save_lora_weights(self, checkpoint_path):
        torch.save(lora.lora_state_dict(self), checkpoint_path)

    def load_pretrained_lora(self, checkpoint_path):
        ck_path = Path(checkpoint_path)
        lora_metadata_path = ck_path.parent.parent / "lora_finetuning_args.json"
        with open(lora_metadata_path, "r") as f:
            metadata = json.load(f)
        
        args = argparse.Namespace(**metadata)
        self.setup_lora_model(args, mark_lora=False)

        self.load_state_dict(torch.load(checkpoint_path), strict=False)
        lora.mark_only_lora_as_trainable(self)

    def replace_aggregator_patch_embed(self):
        for idx, block in enumerate(self.aggregator.patch_embed.blocks):
            if hasattr(self.args, "lora_rank_aggr_patch_embed_qkv") and self.args.lora_rank_aggr_patch_embed_qkv > 0:
                lora_qkv = lora.Linear(
                    block.attn.qkv.in_features, block.attn.qkv.out_features, 
                    r=self.args.lora_rank_aggr_patch_embed_qkv, lora_alpha=self.args.lora_alpha_aggr_patch_embed_qkv
                )
                lora_qkv.weight.data.copy_(block.attn.qkv.weight.data)  
                if block.attn.qkv.bias is not None:
                    lora_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                setattr(self.aggregator.patch_embed.blocks[idx].attn, "qkv", lora_qkv)

            if hasattr(self.args, "lora_rank_aggr_patch_embed_proj") and self.args.lora_rank_aggr_patch_embed_proj > 0:
                lora_proj = lora.Linear(
                    block.attn.proj.in_features, block.attn.proj.out_features, 
                    r=self.args.lora_rank_aggr_patch_embed_proj, lora_alpha=self.args.lora_alpha_aggr_patch_embed_proj
                )
                lora_proj.weight.data.copy_(block.attn.proj.weight.data)  
                if block.attn.proj.bias is not None:
                    lora_proj.bias.data.copy_(block.attn.proj.bias.data)

                setattr(self.aggregator.patch_embed.blocks[idx].attn, "proj", lora_proj)

            if hasattr(self.args, "lora_rank_aggr_patch_embed_mlp") and self.args.lora_rank_aggr_patch_embed_mlp > 0:
                for fc in ["fc1", "fc2"]:
                    mlp_layer = getattr(block.mlp, fc)
                    lora_mlp_fc = lora.Linear(
                        mlp_layer.in_features, mlp_layer.out_features, 
                        r=self.args.lora_rank_aggr_patch_embed_mlp, lora_alpha=self.args.lora_alpha_aggr_patch_embed_mlp
                    )
                    lora_mlp_fc.weight.data.copy_(mlp_layer.weight.data)
                    if mlp_layer.bias is not None:
                        lora_mlp_fc.bias.data.copy_(mlp_layer.bias.data)

                    setattr(self.aggregator.patch_embed.blocks[idx].mlp, fc, lora_mlp_fc)
    
    def replace_aggregator_frame_blocks(self):
        for idx, block in enumerate(self.aggregator.frame_blocks):
            if hasattr(self.args, "lora_rank_aggr_frame_blocks_qkv") and self.args.lora_rank_aggr_frame_blocks_qkv > 0:
                lora_qkv = lora.Linear(
                    block.attn.qkv.in_features, block.attn.qkv.out_features, 
                    r=self.args.lora_rank_aggr_frame_blocks_qkv, lora_alpha=self.args.lora_alpha_aggr_frame_blocks_qkv
                )
                lora_qkv.weight.data.copy_(block.attn.qkv.weight.data)  
                if block.attn.qkv.bias is not None:
                    lora_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                setattr(self.aggregator.frame_blocks[idx].attn, "qkv", lora_qkv)

            if hasattr(self.args, "lora_rank_aggr_frame_blocks_proj") and self.args.lora_rank_aggr_frame_blocks_proj > 0:
                lora_proj = lora.Linear(
                    block.attn.proj.in_features, block.attn.proj.out_features, 
                    r=self.args.lora_rank_aggr_frame_blocks_proj, lora_alpha=self.args.lora_alpha_aggr_frame_blocks_proj
                )
                lora_proj.weight.data.copy_(block.attn.proj.weight.data)  
                if block.attn.proj.bias is not None:
                    lora_proj.bias.data.copy_(block.attn.proj.bias.data)

                setattr(self.aggregator.frame_blocks[idx].attn, "proj", lora_proj)

            if hasattr(self.args, "lora_rank_aggr_frame_blocks_mlp") and self.args.lora_rank_aggr_frame_blocks_mlp > 0:
                for fc in ["fc1", "fc2"]:
                    mlp_layer = getattr(block.mlp, fc)
                    lora_mlp_fc = lora.Linear(
                        mlp_layer.in_features, mlp_layer.out_features, 
                        r=self.args.lora_rank_aggr_frame_blocks_mlp, lora_alpha=self.args.lora_alpha_aggr_frame_blocks_mlp
                    )
                    lora_mlp_fc.weight.data.copy_(mlp_layer.weight.data)
                    if mlp_layer.bias is not None:
                        lora_mlp_fc.bias.data.copy_(mlp_layer.bias.data)

                    setattr(self.aggregator.frame_blocks[idx].mlp, fc, lora_mlp_fc)
    
    def replace_aggregator_global_blocks(self):
        for idx, block in enumerate(self.aggregator.global_blocks):
            if hasattr(self.args, "lora_rank_aggr_global_blocks_qkv") and self.args.lora_rank_aggr_global_blocks_qkv > 0:
                lora_qkv = lora.Linear(
                    block.attn.qkv.in_features, block.attn.qkv.out_features, 
                    r=self.args.lora_rank_aggr_global_blocks_qkv, lora_alpha=self.args.lora_alpha_aggr_global_blocks_qkv
                )
                lora_qkv.weight.data.copy_(block.attn.qkv.weight.data)  
                if block.attn.qkv.bias is not None:
                    lora_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                setattr(self.aggregator.global_blocks[idx].attn, "qkv", lora_qkv)

            if hasattr(self.args, "lora_rank_aggr_global_blocks_proj") and self.args.lora_rank_aggr_global_blocks_proj > 0:
                lora_proj = lora.Linear(
                    block.attn.proj.in_features, block.attn.proj.out_features, 
                    r=self.args.lora_rank_aggr_global_blocks_proj, lora_alpha=self.args.lora_alpha_aggr_global_blocks_proj
                )
                lora_proj.weight.data.copy_(block.attn.proj.weight.data)  
                if block.attn.proj.bias is not None:
                    lora_proj.bias.data.copy_(block.attn.proj.bias.data)

                setattr(self.aggregator.global_blocks[idx].attn, "proj", lora_proj)

            if hasattr(self.args, "lora_rank_aggr_global_blocks_mlp") and self.args.lora_rank_aggr_global_blocks_mlp > 0:
                for fc in ["fc1", "fc2"]:
                    mlp_layer = getattr(block.mlp, fc)
                    lora_mlp_fc = lora.Linear(
                        mlp_layer.in_features, mlp_layer.out_features, 
                        r=self.args.lora_rank_aggr_global_blocks_mlp, lora_alpha=self.args.lora_alpha_aggr_global_blocks_mlp
                    )
                    lora_mlp_fc.weight.data.copy_(mlp_layer.weight.data)
                    if mlp_layer.bias is not None:
                        lora_mlp_fc.bias.data.copy_(mlp_layer.bias.data)

                    setattr(self.aggregator.global_blocks[idx].mlp, fc, lora_mlp_fc)

    def replace_camera_head(self):
        for idx, block in enumerate(self.camera_head.trunk):
            if hasattr(self.args, "lora_rank_camera_head_qkv") and self.args.lora_rank_camera_head_qkv > 0:
                lora_qkv = lora.Linear(
                    block.attn.qkv.in_features, block.attn.qkv.out_features, 
                    r=self.args.lora_rank_camera_head_qkv, lora_alpha=self.args.lora_alpha_camera_head_qkv
                )
                lora_qkv.weight.data.copy_(block.attn.qkv.weight.data)  
                if block.attn.qkv.bias is not None:
                    lora_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                setattr(self.camera_head.trunk[idx].attn, "qkv", lora_qkv)

            if hasattr(self.args, "lora_rank_camera_head_proj") and self.args.lora_rank_camera_head_proj > 0:
                lora_proj = lora.Linear(
                    block.attn.proj.in_features, block.attn.proj.out_features, 
                    r=self.args.lora_rank_camera_head_proj, lora_alpha=self.args.lora_alpha_camera_head_proj
                )
                lora_proj.weight.data.copy_(block.attn.proj.weight.data)  
                if block.attn.proj.bias is not None:
                    lora_proj.bias.data.copy_(block.attn.proj.bias.data)

                setattr(self.camera_head.trunk[idx].attn, "proj", lora_proj)

            if hasattr(self.args, "lora_rank_camera_head_mlp") and self.args.lora_rank_camera_head_mlp > 0:
                for fc in ["fc1", "fc2"]:
                    mlp_layer = getattr(block.mlp, fc)
                    lora_mlp_fc = lora.Linear(
                        mlp_layer.in_features, mlp_layer.out_features, 
                        r=self.args.lora_rank_camera_head_mlp, lora_alpha=self.args.lora_alpha_camera_head_mlp
                    )
                    lora_mlp_fc.weight.data.copy_(mlp_layer.weight.data)
                    if mlp_layer.bias is not None:
                        lora_mlp_fc.bias.data.copy_(mlp_layer.bias.data)

                    setattr(self.camera_head.trunk[idx].mlp, fc, lora_mlp_fc)
