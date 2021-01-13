import torch
import copy


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size,
                                   num_size, hidden_size, USE_CUDA):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices).to(device)
    masked_index = torch.ByteTensor(masked_index).to(device).bool()
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(
        -1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start,
                        unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r
def batch_copy_list(ls):
    copy_ls=[]
    for l in ls:
        copy_ls.append(copy_list(l))
    return copy_ls

def beam_sort(beams,beam_size=5):
    '''
    beams:list of TreeBeam object
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam_num = len(beams)
    out_num = len(beams[0].out)
    scores = []
    for beam in beams:
        scores.append(beam.score)
    scores = torch.stack(scores).to(device)
    topv, topi = scores.topk(beam_size, dim=0)
    #topi=topi.transpose(0,1)
    #topv=topv.transpose(0,1)
    sorted_beams = []
    for beam_idx, batch in enumerate(topi.split(1)):
        # beam.embedding_stack
        # beam.left_childs
        # beam.node_stack
        # beam.out
        # beam.score
        embedding_stack = []
        left_childs = []
        node_stack = []
        out = [[] for _ in range(out_num)]
        score = []
        for b, sort_idx in enumerate(batch.split(1, dim=1)):
            embedding_stack.append(beams[sort_idx].embedding_stack[b])
            left_childs.append(beams[sort_idx].left_childs[b])
            node_stack.append(beams[sort_idx].node_stack[b])
            for out_idx in range(out_num):
                out[out_idx].append(beams[sort_idx].out[out_idx][b])
            score.append(beams[sort_idx].score[b])
        score = torch.stack(score).to(device)
        sorted_beams.append(
            TreeBeam(score, node_stack, embedding_stack, left_childs, out))
    return sorted_beams