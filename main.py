import argparse
import gc
import os
import random
from torch.optim import lr_scheduler, Adam, SGD
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import numpy as np
from model import ParsingNet
from dataloader import DialogueDataset, DiscourseGraph
from tqdm import tqdm
from utils import *

def seed_everything(seed=256):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def trans_structure(predict_parsing_index, prediate_label_index):

    predict_edge_rela = {}
    edge_list = []
    rela_list = []
    assert len(predict_parsing_index)==len(prediate_label_index)
    cur = 1
    for pre_node in predict_parsing_index[1:]:
        pre_node = pre_node - 1
        if pre_node < 0:
            pre_node = 0
        if cur != pre_node:
            edge_list.append((pre_node,cur))
            rela_list.append(prediate_label_index[cur])
        cur += 1
    for index, edge in enumerate(edge_list):
        predict_edge_rela[edge] = rela_list[index]
    return predict_edge_rela
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--glove_vocab_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default='xlnet-base-cased')
    parser.add_argument('--max_vocab_size', type=int, default=1000)
    parser.add_argument('--remake_dataset', action="store_true")
    parser.add_argument('--remake_tokenizer', action="store_true")
    parser.add_argument('--max_edu_dist', type=int, default=20)
    parser.add_argument('--glove_embedding_size', type=int, default=100)
    parser.add_argument('--path_hidden_size', type=int, default=384)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--attention_dropout_DCA', type=float, default=0.1)
    parser.add_argument('--speaker', action='store_true')
    parser.add_argument('--valid_dist', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--pretrained_model_learning_rate', type=float, default=1e-5)
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--pool_size', type=int, default=1)
    parser.add_argument('--eval_pool_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_path', type=str, default='student_model.pt')
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--report_step', type=int, default=50)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--text_max_sep_len', type=int, default= 31)
    parser.add_argument('--total_seq_len', type=int, default= 512)
    parser.add_argument('--seed', type=int, default= 512)
    parser.add_argument('--decoder_input_size', type=int, default= 384)
    parser.add_argument('--decoder_hidden_size', type=int, default= 384)
    parser.add_argument('--classes_label', type=int, default= 17)
    parser.add_argument('--transition_weight', type=int, default= 1)
    parser.add_argument('--graph_weight', type=int, default= 1)
    parser.add_argument('--add_norm', type=bool, default= True)

    args = parser.parse_args()

    seed_everything(args.seed)

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda")

    if not os.path.isdir(args.dataset_dir):
        os.mkdir(args.dataset_dir)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_data_file = os.path.join(args.dataset_dir, 'train.pt')
    eval_data_file = os.path.join(args.dataset_dir, 'eval.pt')
    test_data_file = os.path.join(args.dataset_dir, 'test.pt')

    if os.path.exists(train_data_file) and not args.remake_dataset:
        print('loading dataset..')
        train_dataset = torch.load(train_data_file)
        eval_dataset = torch.load(eval_data_file)
        relations, type2ids, id2types = train_dataset.relations, train_dataset.type2ids, train_dataset.id2types
        print(type2ids)
        if not args.do_train:
            test_dataset = DialogueDataset(args=args, filename=args.test_file, tokenizer=tokenizer, mode='test',text_max_sep_len=args.text_max_sep_len,
                                           total_seq_len = args.total_seq_len)
            test_dataset.get_relations(relations, type2ids, id2types)
            test_dataset.get_discourse_graph()
            torch.save(test_dataset, test_data_file)
    else:
        train_dataset = DialogueDataset(args=args, filename= args.train_file, tokenizer=tokenizer, mode='train',text_max_sep_len=args.text_max_sep_len,
                                           total_seq_len = args.total_seq_len)
        eval_dataset = DialogueDataset(args=args, filename=args.eval_file, tokenizer=tokenizer, mode='eval',text_max_sep_len=args.text_max_sep_len,
                                           total_seq_len = args.total_seq_len)

        relations = train_dataset.relations | eval_dataset.relations
        type2ids, id2types = DialogueDataset.format_relations(relations)
        train_dataset.get_relations(relations, type2ids, id2types)
        eval_dataset.get_relations(relations, type2ids, id2types)

        train_dataset.get_discourse_graph()
        eval_dataset.get_discourse_graph()

        print('saving dataset..')
        torch.save(train_dataset, train_data_file)
        torch.save(eval_dataset, eval_data_file)

    args.relation_type_num = len(id2types)

    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    def train_collate_fn(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[7])
            edu_nums = [x[7] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]
                texts, sep_index, pairs, graphs, lengths, speakers, turns, edu_nums,\
                parsing_index, decoder_input, relation_labels, _ = zip(*batch)

                max_edu_seqlen = max(edu_nums)
                num_batch = len(batch)
                d_inputs = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
                d_outputs = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
                d_output_re = np.zeros([num_batch, max_edu_seqlen], dtype=np.compat.long)
                d_masks = np.zeros([num_batch, max_edu_seqlen, max_edu_seqlen + 1], dtype=np.uint8)
                for batchi,batch_example in enumerate(batch):#当前batch
                    _, _, _, _, _, _, _,_, parsing_index_, decoder_input_, relation_label_,_= batch_example
                    for di, d_input_ in enumerate(decoder_input_):
                        d_inputs[batchi][di] = d_input_
                        d_masks[batchi][di][:d_input_+1] = 1
                    d_outputs[batchi][:len(parsing_index_)] = parsing_index_
                    d_output_re[batchi][:len(relation_label_)] = relation_label_
                texts = torch.stack(texts, dim=0)
                assert texts.shape[0] == len(sep_index)
                lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                d_inputs = torch.from_numpy(d_inputs).long()
                d_outputs = torch.from_numpy(d_outputs).long()
                d_output_re = torch.from_numpy(d_output_re).long()
                d_masks = torch.from_numpy(d_masks).byte()
                yield texts, sep_index, pairs, graphs, lengths, speakers, turns,\
                      edu_nums,d_inputs,d_outputs,d_output_re,d_masks

        return pool(examples)

    def eval_collate_fn(examples):
        texts, sep_index, pairs, graphs, lengths, speakers, turns, edu_nums, \
        parsing_index,decoder_input,relation_labels, ids= zip(*examples)
        max_edu_seqlen = edu_nums[0]
        d_inputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
        d_outputs = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
        d_output_re = np.zeros([1, max_edu_seqlen], dtype=np.compat.long)
        d_masks = np.zeros([1, max_edu_seqlen + 1, max_edu_seqlen + 1], dtype=np.uint8)
        for di, d_input_ in enumerate(decoder_input[0]):
            d_inputs[0][di] = d_input_
            d_masks[0][di][:d_input_ + 1] = 1
        d_outputs[0][:len(parsing_index[0])] = parsing_index[0]
        d_output_re[0][:len(relation_labels[0])] = relation_labels[0]
        lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
        texts = torch.stack(texts, dim=0)
        assert texts.shape[0]==len(sep_index)
        speakers = ints_to_tensor(list(speakers))
        turns = ints_to_tensor(list(turns))
        graphs = ints_to_tensor(list(graphs))
        edu_nums = torch.tensor(edu_nums)
        d_inputs = torch.from_numpy(d_inputs).long()
        d_outputs = torch.from_numpy(d_outputs).long()
        d_output_re = torch.from_numpy(d_output_re).long()
        d_masks = torch.from_numpy(d_masks).byte()
        return texts, sep_index, pairs, graphs, lengths, \
               speakers, turns, edu_nums,   d_inputs,d_outputs,d_output_re,d_masks,ids
    if args.do_train:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.pool_size, shuffle=True,
                                      collate_fn=train_collate_fn)

        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.eval_pool_size, shuffle=False,
                                     collate_fn=eval_collate_fn)
        model = ParsingNet( args, pretrained_model)
        param_groups = [{'params': [param for name, param in model.named_parameters() if
                        name.split('.')[0] != 'pretrained_model'], 'lr': args.learning_rate}]

        param_groups.append({'params': filter(lambda p: p.requires_grad, model.pretrained_model.parameters()),
                             'lr': args.pretrained_model_learning_rate})

        optimizer = AdamW(params=param_groups, lr=args.learning_rate)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma) if args.gamma > 0 else None

        model = model.to(args.device)

        total_step = 0
        eval_result = {}
        accum_train_link_loss, accum_train_label_loss = 0, 0
        accum_distill_loss, accum_classify_loss = 0, 0
        accum_eval_loss = 0
        scheduler_step = 0
        best_eval_result = None
        stop_sign=0

        for epoch in range(args.epoches):
            print('{} epoch training..'.format(epoch + 1))
            print('dialogue model learning rate {:.4f}'.format(optimizer.param_groups[0]['lr']))
            model.train()
            for batch in tqdm(train_dataloader):
                for mini_batch in batch:
                    texts, sep_index_list, pairs, graphs, lengths, speakers, turns, edu_nums,\
                        d_inputs, d_outputs, d_output_re, d_masks = mini_batch
                    texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                    d_masks = d_masks.cuda()
                    d_outputs, d_output_re = d_outputs.cuda(), d_output_re.cuda()
                    grounds = (d_outputs, d_output_re)
                    optimizer.zero_grad()
                    graph_link_loss, graph_label_loss, split_loss, rel_loss = \
                        model.Training_loss_batch(
                            texts,sep_index_list, graphs, lengths, speakers, turns, edu_nums, pairs, \
                            d_inputs, d_masks, grounds)
                    loss = args.transition_weight * (split_loss + rel_loss) + \
                           args.graph_weight * (graph_link_loss + graph_label_loss )
                    accum_train_link_loss += graph_link_loss.item()
                    accum_train_label_loss += graph_label_loss.item()
                    loss.backward()
                    optimizer.step()

                if (total_step + 1) % args.report_step == 0:
                    print(
                        '\t{} step loss: {:.4f} {:.4f}, distill loss: {:.4f}'.format(total_step + 1,
                                                               accum_train_link_loss / args.report_step,
                                                               accum_train_label_loss / args.report_step,
                                                               accum_distill_loss / args.report_step))
                    accum_train_link_loss, accum_train_label_loss, accum_distill_loss = 0, 0, 0
                total_step += 1

                if scheduler and optimizer.param_groups[0]['lr'] > args.min_lr:
                    scheduler.step()
            print('{} epoch training done, begin evaluating..'.format(epoch + 1))
            accum_eval_link_loss, accum_eval_label_loss = [], []

            model = model.eval()

            eval_matrix_transition = {
                'hypothesis': None,
                'reference': None,
                'edu_num': None
            }
            eval_matrix_graph = {
                'hypothesis': None,
                'reference': None,
                'edu_num': None
            }
            for batch in eval_dataloader:
                texts, sep_index_list, pairs, graphs, lengths, speakers, turns, \
                edu_nums, d_inputs, d_outputs, d_output_re, d_masks, ids = batch
                texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                d_inputs, d_outputs, d_output_re, d_masks = d_inputs.cuda(), d_outputs.cuda(), d_output_re.cuda(), d_masks.cuda()
                mask = get_mask(edu_nums + 1, args.max_edu_dist).cuda()

                with torch.no_grad():
                    eval_link_loss, eval_label_loss = model.eval_loss(texts,sep_index_list, graphs, lengths, \
                                                                      speakers, turns, edu_nums, pairs, \
                                                                      d_inputs, d_masks, d_outputs, d_output_re)
                    link, rel, graph_predict_result = model.test(texts, sep_index_list, graphs, lengths, \
                                                                 speakers, turns, edu_nums, pairs, \
                                                                 d_inputs, d_masks, d_outputs, d_output_re)
                    hp_pairs = trans_structure(link, rel)
                    reference = trans_structure(list(d_outputs[0].cpu().numpy()), list(d_output_re[0].cpu().numpy()))
                    predicted_result = {'hypothesis': hp_pairs,
                                        'reference': reference,
                                        'edu_num': edu_nums.cpu()[0].item()}
                    record_eval_result(eval_matrix_transition, predicted_result)
                    record_eval_result(eval_matrix_graph, graph_predict_result)
                accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
                accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

            print("---eval result---")
            a, b = zip(*accum_eval_link_loss)
            c, d = zip(*accum_eval_label_loss)
            eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
            print('eval loss : {:.4f} {:.4f}'.format(eval_link_loss, eval_label_loss))
            eval_loss = eval_link_loss + eval_label_loss
            f1_bi, f1_multi = tsinghua_F1(eval_matrix_graph)
            print("link micro-f1 graph-based test : {}\n"
                  "label micro-f1 graph-based test: {}".format(f1_bi, f1_multi))
            f1_bi, f1_multi = tsinghua_F1(eval_matrix_transition)
            print("link micro-f1 transition-based test : {}\n"
                  "label micro-f1 transition-based test: {}".format(f1_bi, f1_multi))
            stop_sign += 1
            if best_eval_result is None or best_eval_result - eval_loss > 0:
                print(best_eval_result)
                best_eval_result = eval_loss
                stop_sign=0
                if args.save_model:
                    print('saving model..')
                    torch.save(model.state_dict(), args.model_path)
            elif stop_sign+1>args.early_stop:
                break
        for k, v in eval_result.items():
            print(k, v)
    else:
        fw = open('result.json','w',encoding='utf8')

        eval_matrix = {
            'hypothesis': None,
            'reference': None,
            'edu_num': None
        }

        p_dict = {}
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_pool_size, shuffle=False,
                                     collate_fn=eval_collate_fn)

        print('loading model state dict..')
        model = ParsingNet(args, pretrained_model)

        print(args.model_path)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(args.device)
        model = model.eval()

        accum_eval_link_loss, accum_eval_label_loss = [], []
        for batch in test_dataloader:
            texts, sep_index_list, pairs, graphs, lengths, speakers, turns, \
            edu_nums, d_inputs, d_outputs, d_output_re, d_masks, ids = batch
            texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
            d_inputs, d_outputs, d_output_re, d_masks = d_inputs.cuda(), d_outputs.cuda(), d_output_re.cuda(), d_masks.cuda()
            mask = get_mask(edu_nums + 1, args.max_edu_dist).cuda()

            with torch.no_grad():
                eval_link_loss, eval_label_loss = model.eval_loss(texts, sep_index_list, graphs, lengths, \
                                                                  speakers, turns, edu_nums, pairs, \
                                                                  d_inputs, d_masks, d_outputs, d_output_re)
                link, rel, graph_predict_result = model.test(texts, sep_index_list, graphs, lengths, \
                                                             speakers, turns, edu_nums, pairs, \
                                                             d_inputs, d_masks, d_outputs, d_output_re)
                hp_pairs = trans_structure(link, rel)
                reference = trans_structure(list(d_outputs[0].cpu().numpy()), list(d_output_re[0].cpu().numpy()))
                predicted_result = {'hypothesis': hp_pairs,
                                    'reference': reference,
                                    'edu_num': edu_nums.cpu()[0].item()}
                record_eval_result(eval_matrix, graph_predict_result)
            accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
            accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

        print("---eval result---")
        a, b = zip(*accum_eval_link_loss)
        c, d = zip(*accum_eval_label_loss)
        eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
        print('eval loss : {:.4f} {:.4f}'.format(eval_link_loss, eval_label_loss))
        eval_loss = eval_link_loss + eval_label_loss
        f1_bi, f1_multi = tsinghua_F1(eval_matrix)
        print("link micro-f1 graph-based test : {}\n"
              "label micro-f1 graph-based test: {}".format(f1_bi, f1_multi))

