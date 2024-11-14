class Config:
    task_name = 'long_term_forecast'  # 任务类型
    pred_len = 24  # 预测长度
    seq_len = 64  # 输入序列长度（确保T=N以便reshape为正方形图像）
    model_dim = 768  # 模型维度，与ViLT的hidden_size一致
    num_heads = 8  # 注意力头数
    dropout = 0.1  # dropout率
    output_dim = 1  # 输出维度，例如预测单变量
    # VisionTS配置
    vm_arch = 'diffusion_model'  # 更新为diffusion_model
    ft_type = 'finetune'  # VisionTS的微调类型
    vm_pretrained = 1  # 是否使用预训练权重
    vm_ckpt = 'path_to_checkpoint'  # VisionTS的检查点路径（如果有）
    # TimeLLM配置
    llm_model = 'LLAMA'  # TimeLLM使用的LLM模型
    llm_layers = 12  # LLM的层数
    llm_dim = 768  # LLM的维度
    prompt_domain = False  # 是否使用自定义领域提示
    content = '自定义内容描述'  # 如果使用自定义提示，设置相应内容
    d_ff = 2048  # Feedforward层维度
    patch_len = 16  # PatchEmbedding的patch长度
    stride = 8  # PatchEmbedding的步幅
    enc_in = 1  # 输入维度  