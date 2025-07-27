VGGT(
  (aggregator): Aggregator(
    (patch_embed): DinoVisionTransformer(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
        (norm): Identity()
      )
      (blocks): ModuleList(
        (0-23): 24 x NestedTensorBlock(
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (attn): MemEffAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (q_norm): Identity()
            (k_norm): Identity()
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
          )
          (ls1): LayerScale()
          (drop_path1): Identity()
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
          (ls2): LayerScale()
          (drop_path2): Identity()
        )
      )
      (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (head): Identity()
    )
    (rope): RotaryPositionEmbedding2D()
    (frame_blocks): ModuleList(
      (0-23): 24 x Block(
        (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=1024, out_features=3072, bias=True)
          (q_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1024, out_features=1024, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (rope): RotaryPositionEmbedding2D()
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
    (global_blocks): ModuleList(
      (0-23): 24 x Block(
        (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=1024, out_features=3072, bias=True)
          (q_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (k_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=1024, out_features=1024, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (rope): RotaryPositionEmbedding2D()
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
  )
  (camera_head): CameraHead(
    (trunk): Sequential(
      (0): Block(
        (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=True)
          (q_norm): Identity()
          (k_norm): Identity()
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
      (1): Block(
        (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=True)
          (q_norm): Identity()
          (k_norm): Identity()
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
      (2): Block(
        (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=True)
          (q_norm): Identity()
          (k_norm): Identity()
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
      (3): Block(
        (norm1): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (attn): Attention(
          (qkv): Linear(in_features=2048, out_features=6144, bias=True)
          (q_norm): Identity()
          (k_norm): Identity()
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=2048, out_features=2048, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (ls1): LayerScale()
        (drop_path1): Identity()
        (norm2): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (act): GELU(approximate='none')
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
        (ls2): LayerScale()
        (drop_path2): Identity()
      )
    )
    (token_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (trunk_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (embed_pose): Linear(in_features=9, out_features=2048, bias=True)
    (poseLN_modulation): Sequential(
      (0): SiLU()
      (1): Linear(in_features=2048, out_features=6144, bias=True)
    )
    (adaln_norm): LayerNorm((2048,), eps=1e-06, elementwise_affine=False)
    (pose_branch): Mlp(
      (fc1): Linear(in_features=2048, out_features=1024, bias=True)
      (act): GELU(approximate='none')
      (fc2): Linear(in_features=1024, out_features=9, bias=True)
      (drop): Dropout(p=0, inplace=False)
    )
  )
  (point_head): DPTHead(
    (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (projects): ModuleList(
      (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
      (2-3): 2 x Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
    )
    (resize_layers): ModuleList(
      (0): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
      (1): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
      (2): Identity()
      (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (scratch): Module(
      (layer1_rn): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer2_rn): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer3_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer4_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (refinenet1): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet2): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet3): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet4): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (output_conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (output_conv2): Sequential(
        (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (depth_head): DPTHead(
    (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (projects): ModuleList(
      (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
      (2-3): 2 x Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
    )
    (resize_layers): ModuleList(
      (0): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
      (1): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
      (2): Identity()
      (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    (scratch): Module(
      (layer1_rn): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer2_rn): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer3_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (layer4_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (refinenet1): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet2): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet3): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit1): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (refinenet4): FeatureFusionBlock(
        (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (resConfUnit2): ResidualConvUnit(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (activation): ReLU(inplace=True)
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (skip_add): FloatFunctional(
          (activation_post_process): Identity()
        )
      )
      (output_conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (output_conv2): Sequential(
        (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (track_head): TrackHead(
    (feature_extractor): DPTHead(
      (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (projects): ModuleList(
        (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
        (2-3): 2 x Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
      )
      (resize_layers): ModuleList(
        (0): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
        (1): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        (2): Identity()
        (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (scratch): Module(
        (layer1_rn): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer2_rn): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer3_rn): Conv2d(1024, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (layer4_rn): Conv2d(1024, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (refinenet1): FeatureFusionBlock(
          (out_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet2): FeatureFusionBlock(
          (out_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet3): FeatureFusionBlock(
          (out_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit1): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (resConfUnit2): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (refinenet4): FeatureFusionBlock(
          (out_conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (resConfUnit2): ResidualConvUnit(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (activation): ReLU(inplace=True)
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (skip_add): FloatFunctional(
            (activation_post_process): Identity()
          )
        )
        (output_conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (tracker): BaseTrackerPredictor(
      (corr_mlp): Mlp(
        (fc1): Linear(in_features=567, out_features=384, bias=True)
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (fc2): Linear(in_features=384, out_features=128, bias=True)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (updateformer): EfficientUpdateFormer(
        (input_norm): LayerNorm((388,), eps=1e-05, elementwise_affine=True)
        (input_transform): Linear(in_features=388, out_features=384, bias=True)
        (output_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        (flow_head): Linear(in_features=384, out_features=130, bias=True)
        (time_blocks): ModuleList(
          (0-5): 6 x AttnBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0, inplace=False)
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop2): Dropout(p=0, inplace=False)
            )
          )
        )
        (space_virtual_blocks): ModuleList(
          (0-5): 6 x AttnBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0, inplace=False)
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop2): Dropout(p=0, inplace=False)
            )
          )
        )
        (space_point2virtual_blocks): ModuleList(
          (0-5): 6 x CrossAttnBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (cross_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0, inplace=False)
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop2): Dropout(p=0, inplace=False)
            )
          )
        )
        (space_virtual2point_blocks): ModuleList(
          (0-5): 6 x CrossAttnBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm_context): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (cross_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0, inplace=False)
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop2): Dropout(p=0, inplace=False)
            )
          )
        )
      )
      (fmap_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (ffeat_norm): GroupNorm(1, 128, eps=1e-05, affine=True)
      (ffeat_updater): Sequential(
        (0): Linear(in_features=128, out_features=128, bias=True)
        (1): GELU(approximate='none')
      )
      (vis_predictor): Sequential(
        (0): Linear(in_features=128, out_features=1, bias=True)
      )
      (conf_predictor): Sequential(
        (0): Linear(in_features=128, out_features=1, bias=True)
      )
    )
  )
)