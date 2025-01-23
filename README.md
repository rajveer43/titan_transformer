# titan_transformer



TitanModel(
  (embedding): Embedding(50000, 512)
  (transformer): TitanTransformer(
    (layers): ModuleList(
      (0-11): 12 x TitanBlock(
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (attn): TitanAttention(
          (q_proj): Linear(in_features=512, out_features=512, bias=True)
          (k_proj): Linear(in_features=512, out_features=512, bias=True)
          (v_proj): Linear(in_features=512, out_features=512, bias=True)
          (q_conv): DepthwiseSeparableConv1d(
            (depthwise): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), groups=512)
            (pointwise): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          )
          (k_conv): DepthwiseSeparableConv1d(
            (depthwise): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), groups=512)
            (pointwise): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          )
          (v_conv): DepthwiseSeparableConv1d(
            (depthwise): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), groups=512)
            (pointwise): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          )
          (out_proj): Linear(in_features=512, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (0): Linear(in_features=512, out_features=2048, bias=True)
          (1): SiLU()
          (2): GatingMechanism(
            (gate_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (transform_proj): Linear(in_features=2048, out_features=2048, bias=True)
          )
          (3): Dropout(p=0.1, inplace=False)
          (4): Linear(in_features=2048, out_features=512, bias=True)
          (5): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
)
