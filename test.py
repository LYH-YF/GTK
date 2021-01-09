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


class Tester(object):
    def __init__(self):
        super().__init__()
        self.args = config.ARGS
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.build_dataset(config.RANDOM_SEED)
        self.load_model(self.args)

    def load_model(self, args):
        vocab_size = self.data_set.vocab.input_vocab_len
        # model
        self.embedding = BertEmbedding(args, vocab_size).to(self.device)
        self.encoder = KBert_Encoder(args).to(self.device)
        self.predict = Prediction(args).to(self.device)
        self.generate = GenerateNode(args).to(self.device)
        self.merge = Merge(args).to(self.device)

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
        for batch in self.data_set.load_data(1, "test"):
            val_ac, equ_ac = self.test_batch(batch)
            if val_ac:
                self.value_acc += 1
            if equ_ac:
                self.equ_acc += 1
            self.test_total += 1
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

        test_out=self.run_test(batch["question"],batch["ques mask"],batch["num pos"],batch["num mask"],\
                            unk,num_start,batch["position"],batch["visible matrix"],CUDA_USE)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(
            test_out, batch["equation"].tolist()[0], self.data_set.vocab,
            batch["num list"][0], batch["num stack"][0])
        return val_ac, equ_ac

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

        all_node_outputs=self.generate_nodes(encoder_outputs,problem_output,batch_size,padding_hidden,mask,num_mask,num_pos,\
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


if __name__ == "__main__":
    tester = Tester()
    print("start testing...")
    tester.test()
    print("testing finished.")