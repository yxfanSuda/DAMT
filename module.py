import math
import torch
from torch import nn
from utils import get_mask, get_node_mask

class DAMT(nn.Module):
    def __init__(self, params, pretrained_model=None):
        super().__init__()
        self.params = params
        self.pretrained_model = pretrained_model

        self.dialog_gru = nn.GRU(params.hidden_size, params.hidden_size // 2, batch_first=True, bidirectional=True)

        self.path_emb = PathEmbedding(params)
        self.path_update = PathUpdateModel(params)
        self.gnn = StructureAwareAttention(params.hidden_size, params.path_hidden_size, params.num_heads,
                                           params.dropout)

        self.link_classifier = Classifier(params.path_hidden_size * 2, params.path_hidden_size,
                                          1)
        self.label_classifier = Classifier(params.path_hidden_size * 2,
                                           params.path_hidden_size,
                                           params.relation_type_num)
        self.layer_num = params.num_layers
        self.norm = nn.LayerNorm(params.hidden_size)
        self.dropout = nn.Dropout(params.dropout)
        self.hidden_size = params.hidden_size
        self.root = nn.Parameter(torch.zeros(params.hidden_size), requires_grad=False)
        self.G_fc = nn.Linear(params.hidden_size, params.hidden_size)
        self.T_fc = nn.Linear(params.hidden_size, params.hidden_size)
        self.G_T_Emb = SelfAttention(self.G_fc, self.T_fc)
        self.T_block1 = G_T_Block( params.hidden_size)

    def __fetch_sep_rep(self, ten_output, seq_index):
        batch, _, _ = ten_output.shape
        sep_re_list = []
        for index in range(batch):
            cur_seq_index = seq_index[index]
            cur_output = ten_output[index]
            sep_re_list.append(cur_output[cur_seq_index])
        return torch.cat(sep_re_list, dim=0)

    def padding_sep_index_list(self, sep_index_list):

        max_edu = max([len(a) for a in sep_index_list])
        total_new_sep_index_list = []
        for index_list in sep_index_list:
            new_sep_index_list = []
            gap = max_edu - len(index_list)
            new_sep_index_list.extend(index_list)
            for i in range(gap):
                new_sep_index_list.append(index_list[-1])
            total_new_sep_index_list.append(new_sep_index_list)
        return max_edu, total_new_sep_index_list

    def forward(self, texts, sep_index_list, lengths, edu_nums, speakers, turns):
        batch_size = texts.shape[0]
        edu_num, pad_sep_index_list = self.padding_sep_index_list(sep_index_list)
        node_num = edu_num + 1

        sentences = self.pretrained_model(texts)[0]
        sentences = self.__fetch_sep_rep(sentences, pad_sep_index_list)
        nodes = torch.cat((self.root.expand(batch_size, 1, sentences.size(-1)),
                               sentences.reshape(batch_size, edu_num, -1)), dim=1)
        dialog_output_doc_t, dialog_hx_t = self.dialog_gru(nodes)
        dialog_hx_t = dialog_hx_t.contiguous()
        dialog_hx_t = dialog_hx_t[:1, :, :] + dialog_hx_t[1:, :, :]
        mask = get_node_mask(edu_nums + 1, node_num)
        H_G, H_T = self.G_T_Emb(nodes, nodes, mask)
        H_G, H_T = self.T_block1(H_G + nodes, H_T + nodes, mask)
        if not self.params.add_norm:
            G_nodes = H_G + nodes
            T_nodes = H_T + nodes
        else:
            G_nodes = self.norm(H_G + nodes)
            T_nodes = self.norm(H_T + nodes)
        edu_nums = edu_nums + 1
        edu_attn_mask = torch.arange(edu_nums.max()).expand(len(edu_nums), edu_nums.max()).cuda() < edu_nums.unsqueeze(
            1)
        edu_attn_mask = self.gnn.masking_bias(edu_attn_mask)
        const_path = self.path_emb(speakers, turns)
        struct_path = torch.zeros_like(const_path)
        for _ in range(self.layer_num):
            G_nodes = self.gnn(G_nodes, edu_attn_mask, struct_path + const_path)
            struct_path = self.path_update(G_nodes, const_path, struct_path)
            struct_path = self.dropout(struct_path)
        predicted_path = torch.cat((struct_path, struct_path.transpose(1, 2)), -1)
        G_nodes = torch.mean(struct_path, dim=-2)  #
        const_path = self.path_emb(speakers, turns)
        struct_path = torch.zeros_like(const_path)
        for _ in range(self.layer_num):
            T_nodes = self.gnn(T_nodes, edu_attn_mask, struct_path + const_path)
            struct_path = self.path_update(T_nodes, const_path, struct_path)
            struct_path = self.dropout(struct_path)
        T_nodes = torch.mean(struct_path, dim=-2)  #

        return T_nodes, dialog_hx_t, predicted_path,G_nodes

class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_size, path_hidden_size, head_num, dropout):
        super(StructureAwareAttention, self).__init__()
        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.struct_k_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.struct_v_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.o_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.path_norm = nn.LayerNorm(path_hidden_size)

    def forward(self, nodes, bias, paths):
        q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)
        q = self.split_heads(q, self.head_num)
        k = self.split_heads(k, self.head_num)
        v = self.split_heads(v, self.head_num)
        paths = self.path_norm(paths)
        struct_k, struct_v = self.struct_k_transform(paths), self.struct_v_transform(paths)
        q = q * (self.hidden_size // self.head_num) ** -0.5
        w = torch.matmul(q, k.transpose(-1, -2)) + torch.matmul(q.transpose(1, 2),
                                                                struct_k.transpose(-1, -2)).transpose(1, 2) + bias
        w = torch.nn.functional.softmax(w, dim=-1)
        output = torch.matmul(w, v) + torch.matmul(w.transpose(1, 2), struct_v).transpose(1, 2)
        output = self.activation(self.o_transform(self.combine_heads(output)))
        return self.norm(nodes + self.dropout(output))

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]
        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = ~mask * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

class PathEmbedding(nn.Module):
    def __init__(self, params):
        super(PathEmbedding, self).__init__()
        self.speaker = nn.Embedding(2, params.path_hidden_size // 4)
        self.turn = nn.Embedding(2, params.path_hidden_size // 4)
        self.valid_dist = params.valid_dist
        self.position = nn.Embedding(self.valid_dist * 2 + 3, params.path_hidden_size // 2)

        self.tmp = torch.arange(200)
        self.path_pool = self.tmp.expand(200, 200) - self.tmp.unsqueeze(1)
        self.path_pool[self.path_pool > self.valid_dist] = self.valid_dist + 1
        self.path_pool[self.path_pool < -self.valid_dist] = -self.valid_dist - 1
        self.path_pool += self.valid_dist + 1

    def forward(self, speaker, turn):
        batch_size, node_num, _ = speaker.size()
        speaker = self.speaker(speaker)
        turn = self.turn(turn)
        position = self.position(self.path_pool[:node_num, :node_num].cuda())
        position = position.expand(batch_size, node_num, node_num, position.size(-1))
        return torch.cat((speaker, turn, position), dim=-1)

class PathUpdateModel(nn.Module):
    def __init__(self, params):
        super(PathUpdateModel, self).__init__()
        self.x_dim = params.hidden_size
        self.h_dim = params.path_hidden_size
        self.r = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)
        self.z = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)
        self.c = nn.Linear(2*self.x_dim, self.h_dim, True)
        self.u = nn.Linear(self.h_dim, self.h_dim, True)

    def forward(self, nodes, bias, hx, mask=None):
        batch_size, node_num, hidden_size = nodes.size()
        nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        nodes = torch.cat((nodes, nodes.transpose(1, 2)),dim=-1)
        if mask is not None:
            nodes, bias = nodes[mask], bias[mask]
        if hx is None:
            hx = torch.zeros_like(bias)
        rz_input = torch.cat((nodes, hx), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))
        u = torch.tanh(self.c(nodes) + r * self.u(hx))
        new_h = z * hx + (1 - z) * u
        return new_h

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.input_transform = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.output_transform = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        return self.output_transform(self.input_transform(x))

class BiaffineAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels, hidden_size):
        super(BiaffineAttention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W_e = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.W_d = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(num_labels, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.W_e)
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        e_outputs = self.e_mlp(e_outputs)
        d_outputs = self.d_mlp(d_outputs)
        out_e = (self.W_e @ e_outputs.transpose(1, 2)).unsqueeze(2)
        out_d = (self.W_d @ d_outputs.transpose(1, 2)).unsqueeze(3)
        out_u = d_outputs.unsqueeze(1) @ self.U
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)
        out = (out_e + out_d + out_u + self.b).permute(0, 2, 3, 1)
        return out

class SplitAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, hidden_size):
        super(SplitAttention, self).__init__()
        self.biaffine = BiaffineAttention(encoder_size, decoder_size, 1, hidden_size)

    def forward(self, e_outputs, d_outputs, masks):
        biaffine = self.biaffine(e_outputs, d_outputs)
        attn = biaffine.squeeze(-1)
        attn[masks == 0] = -1e8

        return attn

class Decoder(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_dense = nn.Linear(inputs_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_size = hidden_size

    def forward(self, input, state):
        return self.run_step(input, state)

    def run_batch(self, inputs, init_states, masks):
        inputs = self.input_dense(inputs) * masks.unsqueeze(-1).float()
        outputs, _ = self.rnn(inputs, init_states.unsqueeze(0))
        outputs = outputs * masks.unsqueeze(-1).float()
        return outputs

    def run_step(self, input, state):
        input = self.input_dense(input)
        self.rnn.flatten_parameters()
        output, state = self.rnn(input, state)
        return output, state

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class G_T_Block(nn.Module):
    def __init__(self,hidden_size):
        super(G_T_Block, self).__init__()
        self.G_T_Attention = G_T_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)
        self.G_Out = SelfOutput(hidden_size, 0.1)
        self.T_Out = SelfOutput(hidden_size, 0.1)

    def forward(self, H_G_input, H_T_input, mask):
        H_T, H_G = self.G_T_Attention(H_G_input, H_T_input, mask)
        H_T = self.T_Out(H_T, H_T_input)
        H_G = self.G_Out(H_G, H_G_input)
        return H_G, H_T

class SelfAttention(nn.Module):
    def __init__(self, G_emb, T_emb):
        super(SelfAttention, self).__init__()

        self.W_G_emb = G_emb.weight
        self.W_T_emb = T_emb.weight

    def forward(self, input_G, input_T, mask):
        G_score = torch.matmul(input_G, self.W_G_emb.t())
        T_score = torch.matmul(input_T, self.W_T_emb.t())
        G_probs = nn.Softmax(dim=-1)(G_score)
        T_probs = nn.Softmax(dim=-1)(T_score)
        G_res = torch.matmul(G_probs, self.W_G_emb)
        T_res = torch.matmul(T_probs, self.W_T_emb)

        return G_res, T_res

class G_T_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(G_T_SelfAttention, self).__init__()

        self.num_attention_heads = 4
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_T = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_T = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_T = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, G, T, mask):
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(G)
        mixed_key_layer = self.key(T)
        mixed_value_layer = self.value(T)

        mixed_query_layer_T = self.query_T(T)
        mixed_key_layer_T = self.key_T(G)
        mixed_value_layer_T = self.value_T(G)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_T = self.transpose_for_scores(mixed_query_layer_T)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_T = self.transpose_for_scores(mixed_key_layer_T)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_T = self.transpose_for_scores(mixed_value_layer_T)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores_T = torch.matmul(query_layer_T, key_layer_T.transpose(-1, -2))
        attention_scores_T = attention_scores_T / math.sqrt(self.attention_head_size)
        attention_scores_G = attention_scores + attention_mask

        attention_scores_T = attention_scores_T + attention_mask
        attention_probs_T = nn.Softmax(dim=-1)(attention_scores_T)
        attention_probs_G = nn.Softmax(dim=-1)(attention_scores_G)

        attention_probs_T = self.dropout(attention_probs_T)
        attention_probs_G = self.dropout(attention_probs_G)

        context_layer_T = torch.matmul(attention_probs_T, value_layer_T)
        context_layer_G = torch.matmul(attention_probs_G, value_layer)
        context_layer = context_layer_T.permute(0, 2, 1, 3).contiguous()
        context_layer_G = context_layer_G.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_G = context_layer_G.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_G = context_layer_G.view(*new_context_layer_shape_G)
        return context_layer, context_layer_G