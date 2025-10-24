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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        #super表示调用父类的函数，这里的作用是请在当前实例 (self) 的继承链 (MRO) 中，找到紧跟在 LlavaMetaModel 后面的那个类，并调用它的 __init__ 方法
        super(LlavaMetaModel, self).__init__(config)

        #检查配置文件 config 中是否存在 mm_vision_tower 这个属性。如果存在，说明这个模型是一个多模态模型，需要初始化视觉模块
        if hasattr(config, "mm_vision_tower"):
            # 懒加载，实际上只加载了参数文件，需要在后面增加vision_tower.load_model()继续加载模型参数
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            '''
            这是一个针对特定图像处理策略的实现。当 mm_patch_merge_type 配置为 'unpad' 时，意味着模型会处理可变分辨率的图像。
            self.image_newline: 在这种情况下，会创建一个可学习的参数 image_newline。
            这个参数可以被看作是一个特殊的“换行符”嵌入向量，
            用于在拼接不同图像块（patch）的特征时，分隔来自不同行的图像块特征，从而保留图像的空间布局信息。
            '''
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        #防御性编程，用于应对后续可能有多个视觉编码器的复杂情况
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    #加载和配置视觉相关的模块，包括加载预训练权重。
    def initialize_vision_modules(self, model_args, fsdp=None):
        #视觉骨干网络（Visual Tower）模型路径或类型。
        vision_tower = model_args.vision_tower
        #选择视觉模型中用于提取特征的层编号。比如-2表示表示选择视觉模型倒数第二层的输出
        mm_vision_select_layer = model_args.mm_vision_select_layer
        #选择视觉特征的类型（feature type）。
        '''
        视觉模型通常会输出几种不同的特征，例如：
        "patch"：图像分块（patch）特征（常见于 ViT）
        "cls"：特殊的 [CLS] token 特征（代表整张图像的全局语义）
        "mean"：对所有 patch 特征取平均后的全局特征
        功能：
        控制视觉输入的粒度，是使用全图特征还是每个 patch 特征。
        '''
        mm_vision_select_feature = model_args.mm_vision_select_feature
        #预训练的视觉-语言特征对齐层（adapter）的路径或权重。多层感知机（MLP Adapter）
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        #视觉特征合并方式（Patch Merge Type）。
        '''
        在将 patch-level 特征输入语言模型前，常需要进行降维或聚合。
        此参数控制 patch 特征如何被合并或转换。
        常见取值及含义：
        "flat"	将所有 patch 特征直接展开（不合并）输入语言模型。
        "mean"	对所有 patch 特征取平均，形成一个全局视觉向量。
        "mlp"	用一个小型 MLP 进行降维或融合。
        "conv"	使用卷积层进行降采样或特征整合。
        "token"	保留部分重要 patch（如中心区域或注意力高的 patch）。
        '''
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        #如果没有加载视觉编码器，则构建并加载它
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            # fsdp： Fully Sharded Data Parallel 的缩写，中文全称是“完全分片数据并行”。它是 PyTorch 提供的一种先进的、用于大规模模型训练的分布式训练技术。
            if fsdp is not None and len(fsdp) > 0:
                # FSDP 对模型代码的要求
                # 为了让 FSDP 能够正确地“包裹”和“分片”一个模块，这个模块必须被封装在一个 nn.ModuleList 或者 Python 列表中。
                # FSDP 会遍历这个列表，并对其中的每一个 nn.Module 单独应用分片策略。因此会有下面的写法
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        # ... (配置投影层相关的参数)
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                #定义了一个标准差 embed_std，控制初始化范围
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    # 生成一个服从标准正态分布的张量（tensor），即每个元素都来自 N(0,1) 分布
                    # self.config.hidden_size 指定了张量的形状（张量维度，数字个数等） dtype=self.dtype为张量的数据类型
                    # embed_std是缩放因子，可以缩小方差
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            # 这种情况是为了应对 LoRA (Low-Rank Adaptation) 等微调技术。
            # LoRA 可能会冻结模型的大部分原始权重，只训练少量的适配器权重。
            # 如果 mm_projector 之前被冻结了，这行代码确保解冻它，
            # 使得它的参数可以被训练和更新。
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            # 1. 加载权重文件
            # pretrain_mm_mlp_adapter 是一个指向 .bin 或 .pt 文件的路径。
            # torch.load 将这个文件加载到 CPU 内存中，得到一个权重字典。
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            #查找，裁剪，然后返回新的字典对
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # 3. 将解析后的权重加载到模型中
            # load_state_dict 是 PyTorch 模块的标准方法，用于加载权重。
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

#它的核心作用是将在一个正方形画布中被填充（padding）并缩放的、非正方形的图像，恢复到其原始的长宽比。
#裁剪掉因为padding而产生的黑边
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    #将文本向量与语言向量拼接起来
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # type(images) is list: 输入是一个 Python 列表，列表中的每个元素都是一张独立的图像张量。
        # 这对应**“多图对话 (Multi-image conversation)”**的场景。
        # images.ndim == 5: 输入是一个 5D 张量，形状为 [Batch, NumCrops, Channel, Height, Width]。
        # 这对应 LLaVA-NeXT 中提出的**“AnyRes”高分辨率图像处理策略**。一张高分辨率图像被切割成多个小图块（crops），形成一个新的维度。
        # 输入的images一般是已经拆分处理过的图像

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                # x.unsqueeze(dim)在指定的维度上加上一个新的大小为1的维度，比如对于一个形状为(3, 224, 224)的张量，调用x.unsqueeze(0)会得到一个形状为(1, 3, 224, 224)的张量。
                # 确保列表中的每个张量都是4D的 (B, C, H, W)，如果只是 (C, H, W)，则增加一个批次维度
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            # 将所有图像张量（无论是来自 list 还是 5D 张量的切片）在批次维度上拼接成一个大 batch
            #当 images.ndim == 5 时：torch.cat([image for image in images], dim=0) 会把 5D 张量按第 0 维切开成若干 4D 张量，然后在 batch 维（dim=0）重新拼成一个统一的大 batch。
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            # 计算出原始每个图像/图块组的大小，以便后续拆分
            split_sizes = [image.shape[0] for image in images]
            # 将编码后的特征，按照原始大小，拆分回独立的特征组
            image_features = torch.split(image_features, split_sizes, dim=0)
            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            
            if mm_patch_merge_type == 'flat':
                # 功能：把从 start_dim 到 end_dim 的连续维度展平（flatten）成一个维度。返回一个新张量，不改变原张量。
                #就是把一张图片的所有图像块特征，按照它们原有的顺序（通常是从上到下、从左到右），简单地排成一个长长的线性序列。
                image_features = [x.flatten(0, 1) for x in image_features]
            # 空间感知合并，这才是最复杂、最核心的部分，主要服务于 **AnyRes** 高分辨率策略。
            # 将各个图像块的特征重新排列、裁剪、拼接，恢复其空间布局信息
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                # enumerate遍历获取索引和值
                for image_idx, image_feature in enumerate(image_features):
                    #如果大于1（比如我们例子中的7），说明它是高分辨率图，有多个图块（全局+局部）。进入复杂的if分支
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            # 这里需要根据实际的图像块大小，调整图像特征的形状
                            # 将image_feature重新调整形状为 (num_patch_height, num_patch_width, height, width, -1)的维度
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        
                        if 'unpad' in mm_patch_merge_type:
                            # permute() 用于重新排列维度顺序，类似于 numpy 的 transpose()。
                            # permute() 虽然改变了张量的维度顺序，但不会真正改变内存布局。
                            # 这时张量是非连续的（non-contiguous），如果直接调用 .view() 或某些操作，会报错。
                            # 因此 .contiguous() 用于：
                            # 让张量在内存中重新按当前维度顺序连续存储。
                            # 也就是生成一个新的、物理上连续的副本，以保证后续操作安全。
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            # image_feature.flatten(1, 2).flatten(2, 3)连续展平两次，最后的到的是三维的张量（hidden_size，height, width）
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            # 裁剪掉因为padding而产生的黑边
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            # unpad_image 裁剪掉了无效的填充特征，留下了一个“参差不齐”的右边界。虽然我们保留了有效信息，但丢失了重要的结构信息——即图像中一行的特征在哪里结束。
                            # 目的就是在特征层面，为语言模型重新引入这种结构信息。它通过在每一行有效特征的末尾，都拼接上一个特殊的、可学习的“换行符”Token (image_newline)，
                            # 来明确地告诉模型：这一行的视觉信息到此结束了。接下来看到的将是新的一行。
                            # self.model.image_newline[:, None, None]增加换位符，并将其扩展成（hidden_size，1，1）的形状
                            # image_feature.shape[:-1]将image_feature的最后一个维度去除，变成（hidden_size，height）
                            # * 是 解包运算符 ，*tuple 会把元组的内容逐个展开成独立的参数。，例如 t=(3,4); f(*t,1)=f(3,4,1)
                            # (*image_feature.shape[:-1], 1)即为（hidden_size，height，1）
                            # tensor.expand(a, b, c) 的意思是把张量在指定维度上广播成新的形状 (a, b, c)
                            # self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)就变为（hidden_size，height，1）
                            # 最后利用tensor.cat在最后一个维度上拼接，形成（hidden_size，height，width+1）的张量。dim=-1表示在最后一个维度上拼接
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            # 上行代码结束后实现image_feature大小是（hidden_size，height, width+1），多出来的1是换行符特征
                            # .transpose(0, 1)将张量的第0维和第1维交换位置
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                        else:
                            # (num_patch_height, num_patch_width, height, width, -1)经过permute(0, 2, 1, 3, 4)将宽和高集中放在一起
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)

                    # 等于1，说明它是普通分辨率图，没有被进行分割，tensor形状为（1,token_num,每个token的特征维度），只有全局图块特征。进入简单的else分支
                    # 没有被type(images) is list or images.ndim == 5筛除，说明输入图像可能是高分辨率和低分辨率同时输入的图像
                    else:
                        image_feature = image_feature[0]
                        # 只有一张图片，如果有经过pad，就添加换行符
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                # 报错未实现的 mm_patch_merge_type 类型
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        # tune_mm_mlp_adapter: 这个配置项通常意味着用户想要进行参数高效微调 (PEFT)，即只训练模型中一个很小的部分（多模态投影层适配器），冻结其他大部分参数。
        # mm_use_im_start_end: 这个配置项表示用户希望在图像特征序列的前后，显式地插入特殊的**<im_start>和<im_end>**标记。这是一种常见的技术，用于帮助语言模型更好地区分文本和视觉信息。
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # 作者在这里明确表达了设计思想——在复杂的模型中，在每一处都写if x is not None:来处理可选参数是一场噩梦。
        # 因此，不如在最开始就将None值替换为合理的“虚拟”张量，这样后续的所有代码都可以假定这些变量永远是有效的张量，从而变得极其整洁和简单。
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # 根据用户输入的文本（其中包含 <image> 占位符），精确地将图片特征张量和文本张量按正确的顺序穿起来。
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # 经过之前的处理，input_ids 已经是一个列表，其中每个元素 (cur_input_ids) 是一个去掉了填充的、代表一句话的1D张量
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # (cur_input_ids == IMAGE_TOKEN_INDEX)会产生一个布尔张量，True 的位置就是 <image> 所在的位置，.sum就是求true的数量，即求布尔张量和
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # 这种情况可能发生在这一句话中没有写<image>
            if num_images == 0:
                # image_features: 这是一个列表或元组，包含了本次输入所有图片的特征张量。
                # cur_image_idx: 这是一个指针，初始值为0。
                # 作用: 即使文本里没说，模型也知道这次交互是与一张图片相关的。这行代码的作用就是从图片特征列表中，按顺序取出当前应该处理的那张图片的特征。
                # 例如，如果这是批次中的第一个样本，cur_image_idx就是0，它会取出第一张图片的特征。
                cur_image_features = image_features[cur_image_idx]
                #调用模型的 token embedding 层，将输入文本 cur_input_ids（一串 token 的 ID）映射成词向量。
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                
                #但语义上，它代表了：“我拼接了文本嵌入和图像嵌入，只是当前图像部分为空。”
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            #torch.where返回cur_input_ids == IMAGE_TOKEN_INDEX的索引下标的元组，其中第 0 个元素是满足条件的索引tensor
            #[0]即是取这个索引tensor，.tolist()把 tensor 转成 Python 列表
            #最后形成的是[-1，cur_input_ids == IMAGE_TOKEN_INDEX的下标，cur_input_ids的总体长度]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            # 将IMAGE_TOKEN_INDEX切除
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #将切除了IMAGE_TOKEN_INDEX的片段拼接后送入模型
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            
            cur_new_input_embeds = []
            cur_new_labels = []

            #交错插入文本和图像的输入嵌入变量
            #输入从图像的位置切割，产生的文本数是图像数+1
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            #`new_input_embeds` 和 `new_labels` 列表中的每个样本（即每个嵌入张量和标签张量）都会被截断到这个最大长度。这意味着如果序列过长，会丢弃尾部的数据
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # 通过截断过长序列和填充过短序列，确保批次中的所有样本都具有相同的序列长度 max_len，同时生成 attention_mask 和 position_ids，
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        
        # torch.cat() 是在已有维度上拼接；
        # torch.stack() 是新增一个维度后再拼接。
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    # 现代的 Tokenizer（尤其是 Hugging Face 的那种）不仅做分割，还会做词表映射。


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # model_args.mm_use_im_patch_token: 这是一个布尔标志，指示模型是否使用一个特殊的 token 来代表图像的“补丁”（patch）或单个图像特征块
        if model_args.mm_use_im_patch_token:
            # 如果启用，将 DEFAULT_IMAGE_PATCH_TOKEN 这个字符串作为一个新的特殊 token 添加到 Tokenizer 的词汇表中。例如，这个 token 可能在文本中表示一个单独的视觉特征（而非整个图像）。
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            # 每当 Tokenizer 的词汇表大小改变时（添加了新的 token），模型的词嵌入层（input_embeddings 和 output_embeddings）也需要调整大小以匹配新的词汇表。
            # 这行代码会增加嵌入矩阵的行数，为新 token 腾出空间。新 token 的嵌入通常会随机初始化或以特定方式初始化
            self.resize_token_embeddings(len(tokenizer))

        # model_args.mm_use_im_start_end: 这是一个布尔标志，指示模型是否使用 DEFAULT_IM_START_TOKEN 和 DEFAULT_IM_END_TOKEN 来标记整个图像特征序列的开始和结束。
        # 这在将图像特征作为序列插入文本序列时非常有用，可以帮助模型识别图像的边界。
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            # 说明确实有新的token被加入
            if num_new_tokens > 0:
                # 获取模型的输入嵌入矩阵（通常是 nn.Embedding 层的权重）
                input_embeddings = self.get_input_embeddings().weight.data
                # 获取模型的输出嵌入矩阵（通常是用于生成 token 的 nn.Linear 层的权重）
                output_embeddings = self.get_output_embeddings().weight.data

                # 计算现有（非新添加）token 嵌入的平均值。这是一种常见的初始化新 token 嵌入的方法，因为它能让新 token 的嵌入在一个合理的、已学习的向量空间内开始，而不是完全随机初始化。
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                # 新添加的 DEFAULT_IM_START_TOKEN 和 DEFAULT_IM_END_TOKEN 的输入嵌入设置为这个平均值。
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg


            #  配置 MM-MLP 适配器训练策略
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            #  预训练 MM-MLP 适配器权重加载
            if model_args.pretrain_mm_mlp_adapter:
                # model_args.pretrain_mm_mlp_adapter: 这是一个字符串，指示预训练的 MM-MLP 适配器权重的路径。
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                # 确保只有两个新 token 被添加，因为这个分支专门处理 DEFAULT_IM_START_TOKEN 和 DEFAULT_IM_END_TOKEN。
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
