'''
多模态投影层，实现将视觉编辑器生成的图片向量转化为语言模型能理解的图片向量
'''

import torch
import torch.nn as nn
import re

#实现了数学上的恒等函数 (f(x) = x)
#当视觉塔输出的特征维度 (mm_hidden_size) 恰好等于语言模型输入的维度 (hidden_size) 时，
#或者当研究者想要测试完全不使用投影层会带来什么影响时作为占位符使用
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

#因为简单，所以不用单独写一个project.py而是直接写在这里builder.py中间
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # 表示单个线性层
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    # re 是 Python 自带的 正则表达式模块（import re）。
    # re.match(pattern, string) 用来从字符串的 开头 开始匹配指定的正则模式。
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    # 表示恒等投影
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
