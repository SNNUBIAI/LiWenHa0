import copy
import torch
from torch import nn
import collections
import numpy as np
import torch.nn.functional as F
import math
from copy import deepcopy
from torch.autograd import Variable
Hypothesis = collections.namedtuple('Hypothesis', ['value', 'score'])
def clone_module_to_modulelist(module,module_num):
      """"创建一个类将克隆后的module放入modulelist中"""
      return nn.ModuleList([deepcopy(module) for _ in range(module_num)])
class FeedForward(nn.Module):
    """
    两层具有残差网络的前馈神经网络，FNN网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        :param d_model: FFN第一层输入的维度
        :param d_ff: FNN第二层隐藏层输入的维度
        :param dropout: drop比率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 输入数据，形状为(batch_size, input_len, model_dim)
        :return: 输出数据（FloatTensor），形状为(batch_size, input_len, model_dim)
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # return output + x，即为残差网络
        return output  # + x

class LayerNorm(nn.Module):

      def __int__(self,feature,eps=1e-6):
            #feature:self-attention的x的大小
            super(LayerNorm,self).__int__()
            self.a_2 = nn.Parameter(torch.ones(feature))
            self.b_2 = nn.Parameter(torch.zeros(feature))
            self.eps = eps
      def forward(self,x):
             mean=x.mean(-1,keepdim=True)
             std=x.std(-1,keepdim=True)
             return self.a_2*(x-mean)/(std+self.eps)+self.a_2

class FeatEmbedding(nn.Module):
      """"视频特征向量生成器（可以将该函数中的维度减少变成图片向量维度生成器）"""
      def __int__(self,d_feat,d_model,dropout):
            """"FeatEmbedding的初始化
        :param d_feat: per frame dimension（每帧的维度），作为Linear层输入的维度
        :param d_model: 作为Linear层输出的维度"""
            super(FeatEmbedding, self).__init__()
            self.video_embeddings = nn.Sequential(

                  LayerNorm(d_feat),
                  nn.Dropout(p=dropout),
                  nn.Linear(d_feat, d_model)
            )
      def forward(self,x):
            return self.video_embeddings(x)

class SublayerConnection(nn.Module):
      def __int__(self,size,dropout=0.1):
            super(SublayerConnection,self).__init__()#函数初始化
            self.layer_norm = LayerNorm(size)#定义layer_norm
            self.dropout=nn.Dropout(p=dropout)#做dropout层
      def forward(self,x,sublayer):
            #x为self—attention输入
            #
            return self.dropout(self.layer_norm(x+sublayer(x)))
class WordEmbedding(nn.Module):
    """
    把向量构造成d_model维度的词向量，以便后续送入编码器
    """
    def __int__(self,vocab_size,d_model):
      super(WordEmbedding,self).__int__()
      self.d_model = d_model
      # 字典中有vocab_size个词，词向量维度是d_model，每个词将会被映射成d_model维度的向量
      self.embedding = nn.Embedding(vocab_size, d_model)
      self.embed = self.embedding
    def forward(self,x):
          return self.embed(x)*math.sqrt(self.d_model)


def  self_attention(query,key,value,dropout=None,mask=None):
      d_k=query.size(-1)
      scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
      if mask is not None:
            mask.cuda()
            scores = scores.masked_fill(mask == 0, -1e9)
      self_attn=F.softmax(scores,dim=-1)
      if dropout is not None:
            self_attn = dropout(self_attn)
      return torch.matmul(self_attn, value), self_attn
class MultiHeadAttention(nn.Module):
      def __int__(self):
            super(MultiHeadAttention,self).__int__()

      def forward(self, head, d_model, query, key, value, dropout=0.1, mask=None):
            self.attn = None
            assert (d_model % head == 0)
            self.d_k = d_model // head
            self.head = head
            self.d_model = d_model
            #同源Q,K,V做线性变化
            self.linear_query = nn.Linear(d_model, d_model)
            self.linear_key = nn.Linear(d_model, d_model)
            self.linear_value = nn.Linear(d_model, d_model)
            self.linear_out = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(p=dropout)
            self.attn = None

            n_batch = query.size(0)
            #映射并改变维度
            query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
            key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
            value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)


            x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
            x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
            return self.linear_out(x)
class PositionalEncoding(nn.Module):
      def __int__(self,dim,dropout,max_len=5000):
            super(PositionalEncoding,self).__init__()
            if dim % 2 != 0 :
                  raise ValueError("Cannot use sin/cos positional encoding with "
                                   "odd dim (got dim={:d})".format(dim))
            pe = torch.zeros(max_len,dim)
            position = torch.arange(0,max_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0 , dim , 2 , dtype=torch.float)*-(math.log(10000.0) / dim)))

            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.unsqueeze(1)
            self.register_buffer('pe', pe)
            self.drop_out = nn.Dropout(p=dropout)
            self.dim = dim

      def forward(self, emb, step=None):
            #词向量和位置编码拼接并输出
            emb = emb * math.sqrt(self.dim)
            if step is None:
                  emb = emb + self.pe[:emb.size(0)]
            else:
                  emb = emb + self.pe[step]
            emb = self.drop_out(emb)
            return emb
class PositionWiseFeedForward(nn.Module):
      #前向传递FFN函数
      def __int__(self,d_model,d_ff,dropout=0.1):
            super(PositionWiseFeedForward,self).__int__()
            self.w_1 = nn.Linear(d_model,d_ff)
            self.w_2 = nn.Linear(d_ff,d_model)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
            self.dropout_1 = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.dropout_2 = nn.Dropout(dropout)
      def forward(self,x):
            inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
            output = self.dropout_2(self.w_2(inter))
            return output
def sequence_mask(size):
      attn_shape = (1,size,size)
      mask = np.triu(np.ones(attn_shape),k=0).astype('unit8')
      return (torch.from_numpy(mask)==0).cuda()#将numpy类型转换为tensor类型与原张量进行比较传入GPU中
def ser_trg_mask(src,r2l_trg,trg,pad_idx):
      """
        :param src: 编码器的输入
        :param r2l_trg: r2l方向解码器的输入
        :param trg: l2r方向解码器的输入
        :param pad_idx: pad的索引
        :return: trg为None，返回编码器输入的掩码；trg存在，返回编码器和解码器输入的掩码
        """
      #检查src是否为数组当src长度为4执行下列代码，tuple也可以换成list类型
      if isinstance(src, tuple) and len(src) == 4:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)  # 二维特征向量
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)  # 三维特征向量
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)  # 目标检测特征向量
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)  # 目标关系特征向量
      elif isinstance(src, tuple) and len(src) == 3:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)
      elif isinstance(src, tuple) and len(src) == 2:
            src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
      else:
            # 即只有src_image_mask，即二维特征的mask
            src_mask = src_image_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)
      if trg and r2l_trg:#判断解码器是否需要输入编码
            """ trg_mask是填充掩码和序列掩码，&前是填充掩码，&后是通过subsequent_mask函数得到的序列掩码
            其中type_as，是为了让序列掩码和填充掩码的维度一致"""
            trg_mask = (trg != pad_idx).unsqueeze(1) & sequence_mask(trg.size(1)).type_as(src_image_mask.data)
            # r2l_trg的填充掩码
            r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
            # r2l_trg的填充掩码和序列掩码
            r2l_trg_mask = r2l_pad_mask & sequence_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
            # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
      else:
             return src_mask
class EncoderLayer(nn.Module):
   def __int__(self,size,attn,feed_forward,dropout=0.1):
         """

         :param size: d_model
         :param attn: 初始化后的Multi-Head Attention
         :param feed_forward: 初始后Fedd Forward层
         :param dropout:
         :return:
         """

         super(EncoderLayer,self).__int__()
         self.attn = attn
         self.feed_forward=feed_forward
         self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(size, dropout), 2)
   def forward(self,x,mask):
         """
         :param x:
         :param mask:
         :return:
         编码层第二层子层把经过第一层子层处理后的数据first_x与前馈神经网络送入第二个残差网络进行处理得到Encoder层的输出
         """
         first_x = self.sublayer_connection_list[0](x, lambda x_attn: self.attn(x, x, x, mask))
         return self.sublayer_connection_list[1](first_x, self.feed_forward)
class DecoderLayer(nn.Module):
    """
    一层解码Decoder层
    Mask MultiHeadAttention -> Add & Norm -> Multi-Head Attention -> Add & Norm
    -> Feed Forward -> Add & Norm
    """

    def __init__(self, d_model, attn, feed_forward, sublayer_num, dropout=0.1):
          """
          :param d_model: d_model维度
          :param attn: 已经初始化的Multi-Head Attention层
          :param feed_forward: 已经初始化的Feed Forward层
          :param sublayer_num: 解码器内部子层数，如果未来r2l_memory传入有值，则为4层，否则为普通的3层
          :param dropout: drop比率
          """
          super(DecoderLayer, self).__init__()
          self.attn = attn
          self.feed_forward = feed_forward
          self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(d_model, dropout), sublayer_num)
    def forward(self,x,l2r_memory,src_mask,trg_mask,r2l_memory=None,r2l_trg_mask=None):
          """
              :param x: Decoder的输入
              :param l2r_memory: Encoder的输出，作为Multi-Head Attention的K，V值，为从左到右的Encoder的输出
              :param src_mask: 编码器输入的填充掩码
              :param trg_mask: 解码器输入的填充掩码和序列掩码，即对后面单词的掩码
              :param r2l_memory: 从右到左解码器的输出
              :param r2l_trg_mask: 从右到左解码器的输出的填充掩码和序列掩码
              :return: Encoder的输出
          """
            # 把Decoder的输入数据x和经过一个Masked Multi-Head Attention处理后的first_x_attn送入第一个残差网络进行处理得到first_x
          first_x = self.sublayer_connection_list[0](x, lambda first_x_attn: self.attn(x, x, x, trg_mask))
          """"
        把第一层子层得到的first_x和经过一个Multi-Head Attention处理后的second_x_attn（由first_x和Encoder的输出进行自注意力计算）
        送入第二个残差网络进行处理
        """
          second_x = self.sublayer_connection_list[1](first_x,lambda second_x_attn: self.attn(first_x, l2r_memory, l2r_memory,
                                                                                      src_mask))

          """
          解码器第三层子层
          把经过第二层子层处理后的数据second_x与前馈神经网络送入第三个残差网络进行处理得到Decoder层的输出
      
          如果有r2l_memory数据，则还需要经过一层多头注意力计算，也就是说会有四个残差网络
          r2l_memory是让Decoder层多了一层双向编码中从右到左的编码层
          而只要三个残差网络的Decoder层只有从左到右的编码
          """

          if not r2l_memory:
                # 进行从右到左的编码，增加语义信息
                third_x = self.sublayer_connection_list[-2](second_x,
                                                            lambda third_x_attn: self.attn(second_x, r2l_memory, r2l_memory,
                                                                                           r2l_trg_mask))
                return self.sublayer_connection_list[-1](third_x, self.feed_forward)
          else:
                return self.sublayer_connection_list[-1](second_x, self.feed_forward)


class Encoder(nn.Module):
      #构建n层编码层
      def __int__(self,n,encoder_layer):
            super(Encoder,self).__int__()
            self.encoder_layer_list =  clone_module_to_modulelist(encoder_layer, n)
      def forward(self,x,src_mask):
            for encoder_layer in  self.encoder_layer_list:
                  x = encoder_layer(x,src_mask)
            return x
class R2LDecoder(nn.Module):
    """
    n个含有R2L自注意计算的解码层，该解码层只有3个残差网络
    """

    def __init__(self, n_layers, decoder_layer):
        """
        :param n_layers: Decoder层的层数
        :param decoder_layer: 初始化的Decoder层
        """
        super(R2LDecoder, self).__init__()
        self.decoder_layer_list = clone_module_to_modulelist(decoder_layer, n_layers)

    def forward(self, x, memory, src_mask, trg_mask):
        for decoder_layer in self.decoder_layer_list:
            # 没有传入r2l_memory和r2l_trg_mask，默认值为None，即该Decoder只有3个残差网络
            x = decoder_layer(x, memory, src_mask, trg_mask)
        return x
class L2RDecoder(nn.Module):
    """
    n个含有L2R自注意计算的解码层，该解码层有4个残差网络
    """

    def __init__(self, n_layers, decoder_layer):
        """
        :param n_layers: Decoder层的层数
        :param decoder_layer: 初始化的Decoder层
        """
        super(L2RDecoder, self).__init__()
        self.decoder_layer_list = clone_module_to_modulelist(decoder_layer, n_layers)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for decoder_layer in self.decoder_layer_list:
            # 传入r2l_memory和r2l_trg_mask，即修改默认值，Decoder将具有4个残差网络
            x = decoder_layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x
class WordProbGenerator(nn.Module):
    """
    文本生成器，即把Decoder层的输出通过最后一层softmax层变化为词概率
    """

    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词向量维度
        :param vocab_size: 词典大小
        """
        super(WordProbGenerator, self).__init__()
        # 通过线性层的映射，映射成词典大小的维度
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 通过softmax函数对词概率做出估计
        return F.log_softmax(self.linear(x), dim=-1)
class Transformer(nn.Module):
      """"利用上述定义模块完成Transformer的拼接"""
      def __int__(self,vocab,d_model,d_ff,n_heads,n_layer,dropout,feature,device='cuda'):
        """
        :param vocab: 字典长度
        :param d_feat: per frame dimension（每帧的维度）（没用）
        :param d_model: 词向量的长度
        :param d_ff: FNN（FeedForward）第二层隐藏层输入的维度
        :param n_heads: 多头注意力时的头数
        :param n_layers: 编码器和解码器的层数
        :param dropout: drop的比率
        :param feature_mode: 提取视频特征的模式(没用）
        :param device: 是否使用gpu

         """
            super(Transformer,self).__init__()
            self.vocab  = vocab
            self.device = device
            self.feature_model = feature_mode
            attn=MultiHeadAttention(n_heads,d_model,dropout)
            feed_forward = FeedForward(d_model,d_ff)

            # 把特征向量提取成d_model维度的词向量
            self.trg_embed = WordEmbedding(vocab.n_vocabs, d_model)
            # 提取位置向量
            self.pos_embed = PositionalEncoding(d_model, dropout)
            # 编码层
            self.encoder = Encoder(n_layers, EncoderLayer(d_model, deepcopy(attn), deepcopy(feed_forward), dropout))