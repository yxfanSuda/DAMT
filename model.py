import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GraphConvolution, normalize_adj
from utils import get_mask, compute_loss, record_eval_result
from module  import DAMT, Decoder, SplitAttention, BiaffineAttention,Classifier
class ParsingNet(nn.Module):
    def __init__(self, params, pretrained_model):
        super(ParsingNet, self).__init__()
        self.params = params
        self.pretrained_model = pretrained_model
        self.encoder = DAMT(params, self.pretrained_model)
        self.decoder = Decoder(params.decoder_input_size, params.decoder_hidden_size)
        self.splitAttention = SplitAttention(384, 384, 64)
        self.nr_classifier = BiaffineAttention(384, 384, params.classes_label, 128)
        self.activation = torch.nn.ReLU()
        self.layer_norm = nn.LayerNorm(normalized_shape=params.path_hidden_size, elementwise_affine=False)
        self.GCN1 = GraphConvolution(params.path_hidden_size, params.path_hidden_size, act_func=self.activation,
                                     dropout_rate=params.dropout)  # first gcn
        self.link_classifier = Classifier(params.path_hidden_size * 2, params.path_hidden_size, 1)
        self.label_classifier = Classifier(params.path_hidden_size * 2, params.path_hidden_size,
                                           params.classes_label)
        self.DisEmbedding = nn.Embedding(3, 8)
        self.RelEmbedding = nn.Embedding(17, 8)
        self.DisRelLinear = nn.Linear(16, 1)

    def forward(self):
        pass

    def decode2(self,input, memory, state):
        """
        一次解码
        """

        d_input = memory[0,input].unsqueeze(0).unsqueeze(0)
        d_output, state = self.decoder(d_input, state)
        masks = torch.zeros(1, 1, memory.size(1), dtype=torch.uint8)
        masks[0, 0, :input+1] = 1
        masks = masks.cuda()
        split_scores = self.splitAttention(memory, d_output, masks)
        split_scores = split_scores.softmax(dim=-1)
        nr_score = self.nr_classifier(memory, d_output).softmax(dim=-1) * masks.unsqueeze(-1).float()
        split_scores = split_scores[0, 0].cpu().detach().numpy()
        nr_score = nr_score[0, 0].cpu().detach().numpy()
        return split_scores, nr_score, state

    def _decode_batch(self, e_outputs, d_init_states, d_inputs, d_masks):
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)
        d_inputs = e_outputs[torch.arange(e_outputs.size(0)), d_inputs.permute(1, 0)].permute( 1, 0, 2)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()
        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks)
        return d_outputs, d_outputs_masks, d_masks

    def Training_loss_batch(self,input_sentence,sep_index_list, graphs, lengths, \
                            speakers, turns, edu_nums, pairs,\
                            Decoder_Input, Decoder_mask, grounds):
        onehot_structure = torch.zeros(graphs.shape).float().cuda()
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        EncoderOutputs, Last_Hiddenstates, \
        graph_predict_path, GEncoderoutputs= self.encoder(input_sentence,sep_index_list, lengths, edu_nums, speakers, turns)
        Last_Hiddenstates = Last_Hiddenstates.squeeze(0)
        d_outputs, d_outputs_masks, d_masks = self._decode_batch(EncoderOutputs, Last_Hiddenstates, Decoder_Input,Decoder_mask)
        splits_ground, nrs_ground = grounds
        splits_attn = self.splitAttention(EncoderOutputs, d_outputs, d_masks)
        splits_predict_ = splits_attn.log_softmax(dim=2)
        splits_ground_ = splits_ground.view(-1)
        splits_predict = splits_predict_.view(splits_ground_.size(0), -1)
        splits_masks = d_outputs_masks.view(-1).float()
        splits_loss = F.nll_loss(splits_predict, splits_ground_, reduction="none")
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()
        nr_score = self.nr_classifier(EncoderOutputs, d_outputs)
        nr_score = nr_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()
        for ibatch, (link_score_, rel_score_) in enumerate(zip(splits_predict_, nr_score)):
            step = 2
            for link, rel in zip(link_score_.argmax(-1)[1:], rel_score_.argmax(-1)[1:]):
                if link != 0:
                    onehot_structure[ibatch, step, link] = 1.0
                    if abs(step-2-link.cpu().item())<=1:
                        transition_structure[ibatch, step, link] = 1
                    else:
                        transition_structure[ibatch, step, link] = 2
                    transition_rel[ibatch, step, link] = rel[link]
                step += 1
        structure_embedding = self.DisEmbedding(transition_structure.cuda())
        rel_embedding = self.RelEmbedding(transition_rel.cuda())
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding),dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure)
        nr_score = nr_score.view(nr_score.size(0) * nr_score.size(1), nr_score.size(2), nr_score.size(3))
        target_nr_score = nr_score[torch.arange(nr_score.size(0)), splits_ground_]
        target_nr_ground = nrs_ground.view(-1)
        nr_loss = F.nll_loss(target_nr_score, target_nr_ground)

        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)
        graph_predict_path = graph_predict_path + gcn_output

        link_scores = self.link_classifier(graph_predict_path).squeeze(-1)
        label_scores = self.label_classifier(graph_predict_path)
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.params.max_edu_dist).cuda()
        link_loss, label_loss, _ = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask,
                                                            negative=True)
        graph_link_loss = link_loss.mean()
        graph_label_loss = label_loss.mean()
        return graph_link_loss,graph_label_loss,\
               splits_loss, nr_loss

    def eval_loss(self, input_sentence, sep_index_list, graphs, lengths, \
             speakers, turns, edu_nums, pairs, \
             Decoder_Input, Decoder_mask, d_outputs, d_output_re):

        EncoderOutputs, Last_Hiddenstates, \
        graph_predict_path, GEncoderoutputs = self.encoder(input_sentence,sep_index_list, lengths, edu_nums, speakers, turns)
        Last_Hiddenstates = Last_Hiddenstates
        state = Last_Hiddenstates.detach()
        link_list = []
        rela_list = []
        for input in Decoder_Input[0]:
            link_scores, rela_scores, state = self.decode2(input, EncoderOutputs, state)
            predict_link = link_scores.argmax(-1)
            link_list.append(predict_link)
            if predict_link != input.item():
                predict_rela = rela_scores[predict_link].argmax(-1)
                rela_list.append(predict_rela)
            else:
                rela_list.append(0)
        onehot_structure = torch.zeros(graphs.shape).float().cuda()
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        step = 2
        for link,  rel in zip(link_list[1:], rela_list[1:]):
            if link != 0:
                onehot_structure[0, step, link] = 1.0
                if abs(step-2-link.item())<=1:
                    transition_structure[0, step, link] = 1
                else:
                    transition_structure[0, step, link] = 2
                transition_rel[0, step, link] = rel
                step += 1
        structure_embedding = self.DisEmbedding(transition_structure.cuda())
        rel_embedding = self.RelEmbedding(transition_rel.cuda())
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding), dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure).cuda()

        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)
        graph_predict_path = graph_predict_path + gcn_output

        graph_link_scores = self.link_classifier(graph_predict_path).squeeze(-1)
        graph_label_scores = self.label_classifier(graph_predict_path)
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.params.max_edu_dist).cuda()
        eval_link_loss, eval_label_loss = compute_loss(graph_link_scores, graph_label_scores, graphs, mask)
        return eval_link_loss,eval_label_loss

    def test(self, input_sentence, sep_index_list, graphs, lengths, \
                            speakers, turns, edu_nums, pairs,\
                            Decoder_Input, Decoder_mask, d_outputs, d_output_re):
        EncoderOutputs, Last_Hiddenstates, \
        graph_predict_path,GEncoderoutputs= self.encoder(input_sentence, sep_index_list, lengths, edu_nums, speakers, turns)
        Last_Hiddenstates = Last_Hiddenstates
        state = Last_Hiddenstates.detach()
        link_list = []
        rela_list = []
        for input in Decoder_Input[0]:
            link_scores, rela_scores, state = self.decode2(input, EncoderOutputs,state)
            predict_link = link_scores.argmax(-1)
            link_list.append(predict_link)
            if predict_link!=input.item():
                predict_rela = rela_scores[predict_link].argmax(-1)
                rela_list.append(predict_rela)
            else:
                rela_list.append(0)

        onehot_structure = torch.zeros(graphs.shape).float().cuda()
        transition_structure = torch.zeros(graphs.shape).long()
        transition_rel = torch.zeros(graphs.shape).long()
        step = 2
        for link, rel in zip(link_list[1:], rela_list[1:]):
            if link != 0:
                onehot_structure[0, step, link] = 1.0
                if abs(step - 2 - link.item()) <= 1:
                    transition_structure[0, step, link] = 1
                else:
                    transition_structure[0, step, link] = 2
                transition_rel[0, step, link] = rel
                step += 1
        structure_embedding = self.DisEmbedding(transition_structure.cuda())
        rel_embedding = self.RelEmbedding(transition_rel.cuda())
        alpha = self.DisRelLinear(torch.cat((structure_embedding, rel_embedding), dim=-1)).squeeze(-1)
        weight_matrix = alpha.matmul(onehot_structure).cuda()

        gcn1_output = self.GCN1(GEncoderoutputs, weight_matrix)
        gcn_output = gcn1_output
        batch_size, node_num, hidden_size = gcn_output.size()
        gcn_output = gcn_output.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        gcn_output = torch.cat((gcn_output, gcn_output.transpose(1, 2)), dim=-1)
        graph_predict_path = graph_predict_path + gcn_output
        graph_link_scores = self.link_classifier(graph_predict_path).squeeze(-1)
        graph_label_scores = self.label_classifier(graph_predict_path)
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist= self.params.max_edu_dist).cuda()
        batch_size = graph_link_scores.size(0)
        max_len = edu_nums.max()
        graph_link_scores[~mask] = -1e9
        predicted_links = torch.argmax(graph_link_scores, dim=-1)
        predicted_labels = torch.argmax(graph_label_scores.reshape(-1, max_len + 1,  self.params.classes_label)[
                                            torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                -1)].reshape(batch_size, max_len + 1,  self.params.classes_label),
                                        dim=-1)
        predicted_links = predicted_links[:, 1:] - 1
        predicted_labels = predicted_labels[:, 1:]
        hp_pairs = {}
        step = 1
        while step < edu_nums[0]:
            link = predicted_links[0][step].item()
            label = predicted_labels[0][step].item()
            hp_pairs[(link, step)] = label
            step += 1
        graph_predict_result = {'hypothesis': hp_pairs,
                                  'reference': pairs[0],
                                  'edu_num': step}
        return link_list, rela_list, graph_predict_result