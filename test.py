import time
import torch
import warnings
warnings.filterwarnings("ignore")
from pre_data import process_math23k
from dataset import DataSet
from models.Model import *
import config
from crossentropyloss import masked_cross_entropy
from eval import compute_prefix_tree_result
from tools import time_since
def beam_sort(beams):
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    beam_num=len(beams)
    out_num=len(beams[0].out)
    scores=[]
    for beam in beams:
        scores.append(beam.score)
    scores=torch.stack(scores).to(device)
    topv,topi=scores.topk(beam_num,dim=0)
    #topi=topi.transpose(0,1)
    #topv=topv.transpose(0,1)
    sorted_beams=[]
    for beam_idx,batch in enumerate(topi.split(1)):
        # beam.embedding_stack
        # beam.left_childs
        # beam.node_stack
        # beam.out
        # beam.score
        embedding_stack=[]
        left_childs=[]
        node_stack=[]
        out=[[] for _ in range(out_num)]
        score=[]
        for b,sort_idx in enumerate(batch.split(1,dim=1)):
            embedding_stack.append(beams[sort_idx].embedding_stack[b])
            left_childs.append(beams[sort_idx].left_childs[b])
            node_stack.append(beams[sort_idx].node_stack[b])
            for out_idx in range(out_num):
                out[out_idx].append(beams[sort_idx].out[out_idx][b])
            score.append(beams[sort_idx].score[b])
        score=torch.stack(score).to(device)
        sorted_beams.append(TreeBeam(score,node_stack,embedding_stack,left_childs,out))
    return sorted_beams



class Tester(object):
    def __init__(self):
        super().__init__()
        self.args = config.ARGS
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.build_dataset(config.RANDOM_SEED)
        self.load_model(self.args,config.MODEL_PATH)

    def load_model(self, args,path):
        vocab_size = self.data_set.vocab.input_vocab_len
        # model
        self.embedding = BertEmbedding(args, vocab_size).to(self.device)
        self.encoder = KBert_Encoder(args).to(self.device)
        self.predict = Prediction(args).to(self.device)
        self.generate = GenerateNode(args).to(self.device)
        self.merge = Merge(args).to(self.device)
        
        # check_pnt = torch.load(path)
        # # load parameter of model
        # self.embedding.load_state_dict(check_pnt["embedding"])
        # self.encoder.load_state_dict(check_pnt["encoder"])
        # self.predict.load_state_dict(check_pnt["predict"])
        # self.generate.load_state_dict(check_pnt["generate"])
        # self.merge.load_state_dict(check_pnt["merge"])
        

    def build_dataset(self, random_seed):
        # data processing
        processed_datas, kg, temp_g, copy_nums = process_math23k()
        # build dataset and vocab
        self.data_set = DataSet(processed_datas, self.device, random_seed)
        self.data_set.build_vocab(kg.segment_vocab, temp_g, copy_nums)
        self.args.input_size = len(temp_g)
        self.args.op_nums = self.data_set.vocab.op_nums

    def train2eval(self):
        self.embedding.eval()
        self.encoder.eval()
        self.predict.eval()
        self.generate.eval()
        self.merge.eval()

    def test(self):
        self.value_acc = 0
        self.equ_acc = 0
        self.test_total = 0
        test_time = time.time()
        self.train2eval()
        for batch in self.data_set.load_data(self.args.batch_size, "test"):
            batch_val_ac, batch_equ_ac = self.test_batch(batch)
            
            self.value_acc += batch_val_ac.count(True)
            self.equ_acc += batch_equ_ac.count(True)
            self.test_total += len(batch_val_ac)
        print("test running time:{}".format(time_since(time.time() -
                                                       test_time)))
        print(
            "test total nums:%5d | value accurate nums:%5d | euqation accurate nums:%5d"
            % (self.test_total, self.value_acc, self.equ_acc))
        print(
            "value acc:%1.4f | euqation acc:%1.4f" %
            (self.value_acc / self.test_total, self.equ_acc / self.test_total))

    def test_batch(self, batch):
        unk = self.data_set.vocab.output_dict["UNK"]
        num_start = self.data_set.vocab.num_start
        CUDA_USE = True if torch.cuda.is_available() else False

        test_out=self.run_test(batch["question"],
                                batch["ques mask"],
                                batch["num pos"],
                                batch["num mask"],
                                num_start,
                                batch["position"],
                                batch["visible matrix"],
                                CUDA_USE)
        batch_val_acc=[]
        batch_equ_acc=[]
        for i in range(len(test_out)):
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                test_out[i], batch["equation"].tolist()[i], self.data_set.vocab,
                batch["num list"][i], batch["num stack"][i])
            batch_val_acc.append(val_ac)
            batch_equ_acc.append(equ_ac)
        return batch_val_acc,batch_equ_acc

    def run_test(self,
                 src,
                 mask,
                 num_pos,
                 num_mask,
                 num_start,
                 pos=None,
                 vm=None,
                 USE_CUDA=False):
        emb = self.embedding(src, mask, pos)
        encoder_outputs = self.encoder(
            emb, mask, vm)  #[batch_size x seq_length x hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        problem_output = encoder_outputs[-1, :, :]

        batch_size = problem_output.size(0)
        padding_hidden = torch.FloatTensor(
            [0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0)

        all_node_outputs=self.generate_nodes__(encoder_outputs,problem_output,batch_size,padding_hidden,mask,num_mask,num_pos,\
                                                num_start,USE_CUDA)
        return all_node_outputs
    def generate_nodes(self,encoder_outputs,problem_output,batch_size,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,USE_CUDA,beam_size=5,max_length=MAX_OUTPUT_LENGTH):
        # Prepare input and output variables
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
                        generate_input = torch.LongTensor([out_token
                                                           ]).to(self.device)
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
    def generate_nodes__(self,encoder_outputs,problem_output,batch_size,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,USE_CUDA,beam_size=5,max_length=MAX_OUTPUT_LENGTH):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(
            encoder_outputs, num_pos, batch_size, num_size,
            self.encoder.hidden_size, USE_CUDA)

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        out=[[] for _ in range(batch_size)]
        scores=[0.0 for _ in range(batch_size)]
        beams = [
            TreeBeam(scores, node_stacks, embeddings_stacks, left_childs, [])
        ]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                # if len(b.node_stack[0]) == 0:
                #     current_beams.append(b)
                #     continue
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

                    ti=ti.squeeze()
                    is_op=ti<num_start
                    gen_input=ti*(is_op.float())

                    out_token = ti.tolist()
                    current_out.append(out_token)

                    generate_input = gen_input.long()
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = self.generate(
                        current_embeddings, generate_input,
                        current_context)
                    for idx in range(batch_size):
                        if current_node_stack[idx]==[]:
                            #current_beams.append(b)
                            current_embeddings_stacks[idx].append(b.embedding_stack[idx])
                            current_left_childs.append(b.left_childs[idx])
                            #current_node_stack[idx]=b
                            continue
                        node = current_node_stack[idx].pop()

                        if out_token[idx] < num_start:

                            current_node_stack[idx].append(TreeNode(right_child[idx].unsqueeze(0)))
                            current_node_stack[idx].append(
                                TreeNode(left_child[idx].unsqueeze(0), left_flag=True))

                            current_embeddings_stacks[idx].append(
                                TreeEmbedding(node_label[idx].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[
                                idx, out_token[idx] - num_start].unsqueeze(0)

                            while len(
                                    current_embeddings_stacks[idx]
                            ) > 0 and current_embeddings_stacks[idx][-1].terminal:
                                sub_stree = current_embeddings_stacks[idx].pop()
                                op = current_embeddings_stacks[idx].pop()
                                current_num = self.merge(op.embedding,
                                                        sub_stree.embedding,
                                                        current_num)
                            current_embeddings_stacks[idx].append(
                                TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[idx]
                            ) > 0 and current_embeddings_stacks[idx][-1].terminal:
                            current_left_childs.append(
                                current_embeddings_stacks[idx][-1].embedding)
                        else:
                            current_left_childs.append(None)
                    x1=torch.tensor(b.score)
                    x2=tv.squeeze()
                    current_beams.append(
                        TreeBeam(torch.tensor(b.score) + tv.squeeze(), current_node_stack,
                                current_embeddings_stacks,
                                current_left_childs, current_out))
            #beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams=beam_sort(current_beams)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        all_out_node=beams[0].out
        all_node_outputs=torch.tensor(all_out_node).to(self.device).long().transpose(0,1)
        return all_node_outputs
    def generate_nodes_(self,encoder_outputs,problem_output,batch_size,padding_hidden,seq_mask,num_mask,num_pos,\
                        num_start,USE_CUDA,beam_size=5,max_length=MAX_OUTPUT_LENGTH):
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        #max_target_length = max(target_length)

        all_node_outputs = []
        # all_leafs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(
            encoder_outputs, num_pos, batch_size, num_size,
            self.encoder.hidden_size, USE_CUDA)
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        beams = [
            TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])
        ]
        for t in range(max_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                node_stacks, left_childs, encoder_outputs,
                all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
            # all_leafs.append(p_leaf)
            out_score = nn.functional.log_softmax(torch.cat(
                (op, num_score), dim=1),dim=1)
            topv, topi = out_score.topk(beam_size)
            for v,i in zip(topv.split(1,dim=1),topi.split(1,dim=1)):
                i=i.reshape(batch_size)
                is_op=i<num_start
                gen_input=i*is_op

                out_token = i.tolist()
                all_node_outputs.append(out_token)

                # generate_input = torch.LongTensor(gen_input
                #                                         ).to(self.device)
                generate_input=gen_input.long().squeeze()
                if USE_CUDA:
                    generate_input = generate_input.cuda()
                left_child, right_child, node_label = self.generate(
                    current_embeddings, generate_input,
                    current_context)
                
                for idx,l_c,r_c,o_t,node_stack,o in zip(range(batch_size),left_child.split(1),right_child.split(1),out_token,node_stacks,embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue

                    if o_t < num_start:
                        
                        node_stack.append(TreeNode(r_c))
                        node_stack.append(TreeNode(l_c, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0),
                                            False))
                    else:
                        current_num = current_nums_embeddings[
                        idx, o_t - num_start].unsqueeze(0)
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
            all_node_outputs=torch.tensor(all_node_outputs).to(self.device).long().transpose(0,1)
        return all_node_outputs


if __name__ == "__main__":
    # x=torch.rand((5,30))
    # v,i=x.topk(5,dim=0)
    # print(1)
    tester = Tester()
    print("start testing...")
    tester.test()
    print("testing finished.")