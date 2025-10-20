#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

#保存模型的 超参数（例如隐藏层大小、注意力头数、词汇表大小等）。
#提供 默认初始化值，方便用户快速创建模型
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"



class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

'''
类名称 (Class Name)	主要作用	继承关系/构成	核心功能
LlamaModel	基础语言模型	PreTrainedModel	纯文本处理，输出隐藏状态[1]
LlavaMetaModel	多模态组件管理器	-	管理视觉编码器和投影器[2]
LlavaLlamaModel	融合视觉和语言的核心模型	LlavaMetaModel, LlamaModel	整合图像特征和文本特征[2][5]
LlavaMetaForCausalLM	多模态文本生成的抽象逻辑	-	定义如何准备图像和文本混合输入[2]
LlavaLlamaForCausalLM	完整的、可执行的多模态生成模型	LlamaForCausalLM, LlavaMetaForCausalLM	接收图像和文本，并生成连贯的文本回答[2][5]
'''

'''
当用户输入一张图片和一个问题时，LlavaLlamaForCausalLM 模型开始工作。
它内部的 LlavaMetaModel 部分会调用视觉编码器处理图像，并通过投影器将其转换为与文本嵌入兼容的特征。
接着，LlavaMetaForCausalLM 的逻辑会将这些图像特征与用户问题的文本 token 拼接在一起。
最后，这个拼接好的序列被送入继承自 LlamaForCausalLM 的语言模型部分，模型会像处理纯文本一样，逐字生成答案，完成一次图文对话。
'''

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        #用一个“多模态核心” (LlavaLlamaModel) 替换了原来语言模型的“纯文本核心” (LlamaModel)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # 用于训练，有梯度计算
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # self.prepare_inputs_labels_for_multimodal() 方法
        # 将图像 images 通过视觉编码器和投影器，转换成一系列向量 (image embeddings)。
        # 将文本 input_ids 转换成文本向量 (text embeddings)。
        # 将图像向量和文本向量智能地拼接在一起，形成一个统一的、混合了视觉和语言信息的 inputs_embeds。
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    #用于推理，没有梯度计算
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    #自定义的 prepare_inputs_for_generation 方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # 调用父类的同名方法，获取基础的输入字典
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # 将图像信息重新添加回输入字典
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
