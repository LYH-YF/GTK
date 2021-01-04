import torch
from torch import nn

from models.Sublayer import TransformerLayer, TreeAttn, Score, LayerNorm
from models.utils import *

MAX_OUTPUT_LENGTH = 30


class GTK_Model(nn.Module):
    def __init__(self, embedding, encoder, predict, generate, merge):
        super(GTK_Model, self).__init__()
        '''
        encoder module
        predict module
        generate module
        merge module
        '''
        self.embedding = embedding
        self.encoder = encoder
        self.predict = predict
        self.generate = generate
        self.merge = merge

    def forward(self, src, mask,num_pos,num_mask,unk,num_start,\
                pos=None, vm=None,target_length=None,target=None,nums_stack_batch=None,USE_CUDA=False):
        # if src.size(1) > 128:
        #     print(1)
        #     return None
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb = self.embedding(src, mask, pos)
        encoder_outputs = self.encoder(
            emb, mask, vm)  #[batch_size x seq_length x hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        problem_output = encoder_outputs[-1, :, :]

        batch_size = problem_output.size(0)
        padding_hidden = torch.FloatTensor(
            [0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0)
        if target is not None:
            #target = torch.LongTensor(target).transpose(0, 1)
            target=target.transpose(0,1)
            all_node_outputs=self.generate_nodes(encoder_outputs,problem_output,batch_size,padding_hidden,\
                                                    num_pos,target_length,mask,num_mask,\
                                                    target,nums_stack_batch,unk,num_start,USE_CUDA)
        else:
            all_node_outputs=self.generate_nodes_2(encoder_outputs,problem_output,batch_size,padding_hidden,mask,num_mask,num_pos,\
                                                    num_start,USE_CUDA)
            return all_node_outputs
        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()
        if USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
        return all_node_outputs
    def generate_nodes(self,encoder_outputs,problem_output,batch_size,padding_hidden,num_pos,target_length,seq_mask,num_mask,\
                        target,nums_stack_batch,unk,num_start,USE_CUDA):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(
            encoder_outputs, num_pos, batch_size, num_size,
            self.encoder.hidden_size, USE_CUDA)
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                node_stacks, left_childs, encoder_outputs,
                all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = generate_tree_input(
                target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = self.generate(
                current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size),
                                                   left_child.split(1),
                                                   right_child.split(1),
                                                   node_stacks,
                                                   target[t].tolist(),
                                                   embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0),
                                           False))
                else:
                    current_num = current_nums_embeddings[
                        idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding,
                                                 sub_stree.embedding,
                                                 current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        return all_node_outputs
    def generate_nodes_2(self,encoder_outputs,problem_output,batch_size,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,USE_CUDA,beam_size=5,max_length=MAX_OUTPUT_LENGTH):
        # Prepare input and output variables
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos[0])
        all_nums_encoder_outputs = get_all_number_encoder_outputs(
            encoder_outputs, num_pos, batch_size, num_size,
            self.encoder.hidden_size, USE_CUDA)

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        beams = [
            TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])
        ]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b.node_stack, left_childs, encoder_outputs,
                    all_nums_encoder_outputs, padding_hidden, seq_mask,
                    num_mask)

                out_score = nn.functional.log_softmax(torch.cat(
                    (op, num_score), dim=1),
                                                      dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token]).to(device)
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(
                            current_embeddings, generate_input,
                            current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(
                            TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(
                            TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[
                            0, out_token - num_start].unsqueeze(0)

                        while len(
                                current_embeddings_stacks[0]
                        ) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding,
                                                     sub_stree.embedding,
                                                     current_num)
                        current_embeddings_stacks[0].append(
                            TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]
                           ) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(
                            current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(
                        TreeBeam(b.score + float(tv), current_node_stack,
                                 current_embeddings_stacks,
                                 current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        return beams[0].out


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.embedding_size)
        self.position_embedding = nn.Embedding(self.max_length,
                                               args.embedding_size)
        self.segment_embedding = nn.Embedding(3, args.embedding_size)
        self.layer_norm = LayerNorm(args.embedding_size)

    def forward(self, src, seg, pos=None):
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
                                            dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class KBert_Encoder(nn.Module):
    def __init__(self, args):
        super(KBert_Encoder, self).__init__()
        self.layers_num = args.layers_num
        self.hidden_size = args.hidden_size
        self.transformer = nn.ModuleList(
            [TransformerLayer(args) for _ in range(self.layers_num)])

    def forward(self, emb, seg, vm=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if vm is None:
            mask = (seg > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vm.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, args):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = args.hidden_size
        self.input_size = args.input_size
        self.op_nums = args.op_nums

        # Define layers
        self.dropout = nn.Dropout(args.dropout)

        self.embedding_weight = nn.Parameter(
            torch.randn(1, args.input_size, args.hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(args.hidden_size, args.hidden_size)
        self.concat_r = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.concat_lg = nn.Linear(args.hidden_size, args.hidden_size)
        self.concat_rg = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.ops = nn.Linear(args.hidden_size * 2, args.op_nums)

        self.attn = TreeAttn(args.hidden_size, args.hidden_size)
        self.score = Score(args.hidden_size * 2, args.hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades,
                padding_hidden, seq_mask, mask_nums):
        current_embeddings = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            c=c.to(device)
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                l=l.to(device)
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)
        seq_mask=seq_mask.bool()
        current_attn = self.attn(current_embeddings.transpose(0, 1),
                                 encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(
            0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(
            *repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades),
                                     dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_,
                               mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, args):
        super(GenerateNode, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size

        self.embeddings = nn.Embedding(args.op_nums, args.embedding_size)
        self.em_dropout = nn.Dropout(args.dropout)
        self.generate_l = nn.Linear(args.hidden_size * 2 + args.embedding_size,
                                    args.hidden_size)
        self.generate_r = nn.Linear(args.hidden_size * 2 + args.embedding_size,
                                    args.hidden_size)
        self.generate_lg = nn.Linear(
            args.hidden_size * 2 + args.embedding_size, args.hidden_size)
        self.generate_rg = nn.Linear(
            args.hidden_size * 2 + args.embedding_size, args.hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(
            self.generate_l(
                torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(
            self.generate_lg(
                torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(
            self.generate_r(
                torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(
            self.generate_rg(
                torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, args):
        super(Merge, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size

        self.em_dropout = nn.Dropout(args.dropout)
        self.merge = nn.Linear(args.hidden_size * 2 + args.embedding_size,
                               args.hidden_size)
        self.merge_g = nn.Linear(args.hidden_size * 2 + args.embedding_size,
                                 args.hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(
            self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(
            self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2),
                                   1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


def build_model(args, vocab_size, device):
    emb_module = BertEmbedding(args, vocab_size)
    encoder_module = KBert_Encoder(args)
    pred_module = Prediction(args)
    gen_module = GenerateNode(args)
    merge_module = Merge(args)

    gtk_model = GTK_Model(emb_module, encoder_module, pred_module, gen_module,
                          merge_module).to(device)

    return gtk_model
