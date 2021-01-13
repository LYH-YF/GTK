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


class Trainer(object):
    '''
    structure of Trainer:
        module:
            embedding
            encoder
            predict
            generate
            merge
        optimizer:
            embedding_optimizer
            encoder_optimizer
            predict_optimizer
            generate_optimizer
            merge_optimizer
        scheduler:
            embedding_scheduler
            encoder_scheduler
            predict_scheduler
            generate_scheduler
            merge_scheduler
        train process:
            batch_nums
            batch_i
            start_epoch
            epoch_i
            best_value_acc
            best_equ_acc
        dataset
        args
        device
    '''
    def __init__(self):
        super().__init__()
        self.args = config.ARGS
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.build_dataset(config.RANDOM_SEED)
        self.init_model(self.args, self.data_set.vocab.input_vocab_len)

    def init_model(self, args, vocab_size):
        # model
        self.embedding = BertEmbedding(args, vocab_size).to(self.device)
        self.encoder = KBert_Encoder(args).to(self.device)
        self.predict = Prediction(args).to(self.device)
        self.generate = GenerateNode(args).to(self.device)
        self.merge = Merge(args).to(self.device)
        # optimizer
        self.embedding_optimizer = torch.optim.Adam(
            self.embedding.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        self.predict_optimizer = torch.optim.Adam(
            self.predict.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        self.generate_optimizer = torch.optim.Adam(
            self.generate.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        self.merge_optimizer = torch.optim.Adam(self.merge.parameters(),
                                                lr=args.learning_rate,
                                                weight_decay=args.weight_decay)
        # scheduler
        self.embedding_scheduler = torch.optim.lr_scheduler.StepLR(
            self.embedding_optimizer, step_size=20, gamma=0.5)
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_optimizer, step_size=20, gamma=0.5)
        self.predict_scheduler = torch.optim.lr_scheduler.StepLR(
            self.predict_optimizer, step_size=20, gamma=0.5)
        self.generate_scheduler = torch.optim.lr_scheduler.StepLR(
            self.generate_optimizer, step_size=20, gamma=0.5)
        self.merge_scheduler = torch.optim.lr_scheduler.StepLR(
            self.merge_optimizer, step_size=20, gamma=0.5)
        # load trained model or not
        if args.resume:
            self.load_model(config.MODEL_PATH)
        else:
            self.start_epoch = 0
            self.best_value_acc = 0
            self.best_equ_acc = 0

    def load_model(self, path):
        check_pnt = torch.load(path)
        # load parameter of model
        self.embedding.load_state_dict(check_pnt["embedding"])
        self.encoder.load_state_dict(check_pnt["encoder"])
        self.predict.load_state_dict(check_pnt["predict"])
        self.generate.load_state_dict(check_pnt["generate"])
        self.merge.load_state_dict(check_pnt["merge"])
        # load parameter of optimizer
        self.embedding_optimizer.load_state_dict(
            check_pnt["embedding_optimizer"])
        self.encoder_optimizer.load_state_dict(check_pnt["encoder_optimizer"])
        self.predict_optimizer.load_state_dict(check_pnt["predict_optimizer"])
        self.generate_optimizer.load_state_dict(
            check_pnt["generate_optimizer"])
        self.merge_optimizer.load_state_dict(check_pnt["merge_optimizer"])
        #load parameter of scheduler
        self.embedding_scheduler.load_state_dict(
            check_pnt["embedding_scheduler"])
        self.encoder_scheduler.load_state_dict(check_pnt["encoder_scheduler"])
        self.predict_scheduler.load_state_dict(check_pnt["predict_optimizer"])
        self.generate_scheduler.load_state_dict(
            check_pnt["generate_scheduler"])
        self.merge_scheduler.load_state_dict(check_pnt["merge_scheduler"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_value_acc = check_pnt["value_acc"]
        self.best_equ_acc = check_pnt["equ_acc"]

    def save_model(self, path):
        check_pnt = {
            "embedding": self.embedding.state_dict(),
            "encoder": self.encoder.state_dict(),
            "predict": self.predict.state_dict(),
            "generate": self.generate.state_dict(),
            "merge": self.merge.state_dict(),
            "embedding_optimizer": self.embedding_optimizer.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "predict_optimizer": self.predict_optimizer.state_dict(),
            "generate_optimizer": self.generate_optimizer.state_dict(),
            "merge_optimizer": self.merge_optimizer.state_dict(),
            "embedding_scheduler": self.embedding_scheduler.state_dict(),
            "encoder_scheduler": self.encoder_scheduler.state_dict(),
            "predict_optimizer": self.predict_optimizer.state_dict(),
            "generate_scheduler": self.generate_scheduler.state_dict(),
            "merge_scheduler": self.merge_scheduler.state_dict(),
            "start_epoch": self.epoch_i,
            "value_acc": self.best_value_acc,
            "equ_acc": self.best_equ_acc
        }
        torch.save(check_pnt, path)

    def build_dataset(self, random_seed):
        # data processing
        processed_datas, kg, temp_g, copy_nums = process_math23k()
        # build dataset and vocab
        self.data_set = DataSet(processed_datas, self.device, random_seed)
        self.data_set.build_vocab(kg.segment_vocab, temp_g, copy_nums)
        self.args.input_size = len(temp_g)
        self.args.op_nums = self.data_set.vocab.op_nums

    def eval2train(self):
        self.embedding.train()
        self.encoder.train()
        self.predict.train()
        self.generate.train()
        self.merge.train()

    def train2eval(self):
        self.embedding.eval()
        self.encoder.eval()
        self.predict.eval()
        self.generate.eval()
        self.merge.eval()

    def model_zero_grad(self):
        self.embedding.zero_grad()
        self.encoder.zero_grad()
        self.predict.zero_grad()
        self.generate.zero_grad()
        self.merge.zero_grad()

    def scheduler_step(self):
        self.embedding_scheduler.step()
        self.encoder_scheduler.step()
        self.predict_scheduler.step()
        self.generate_scheduler.step()
        self.merge_scheduler.step()

    def optimizer_step(self):
        self.embedding_optimizer.step()
        self.encoder_optimizer.step()
        self.predict_optimizer.step()
        self.generate_optimizer.step()
        self.merge_optimizer.step()
    def run_train(self, src, mask,num_pos,num_mask,unk,num_start,\
                    pos=None, vm=None,target_length=None,target=None,nums_stack_batch=None,USE_CUDA=False):
        emb = self.embedding(src, mask, pos)
        encoder_outputs = self.encoder(
            emb, mask, vm)  #[batch_size x seq_length x hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        problem_output = encoder_outputs[-1, :, :]

        batch_size = problem_output.size(0)
        padding_hidden = torch.FloatTensor(
            [0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0)
        if target is not None:
            target = target.transpose(0, 1)
            all_node_outputs=self.generate_nodes(encoder_outputs,problem_output,batch_size,padding_hidden,\
                                                    num_pos,target_length,mask,num_mask,\
                                                    target,nums_stack_batch,unk,num_start,USE_CUDA)
        else:
            all_node_outputs=self.generate_nodes_2(encoder_outputs,problem_output,batch_size,padding_hidden,mask,num_mask,num_pos,\
                                                    num_start,USE_CUDA)
            return all_node_outputs
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

        target = target.transpose(0, 1).contiguous()
        if USE_CUDA:
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
        return all_node_outputs

    def train_batch(self, batch):
        unk = self.data_set.vocab.output_dict["UNK"]
        num_start = self.data_set.vocab.num_start
        CUDA_USE = True if torch.cuda.is_available() else False

        self.model_zero_grad()
        outputs=self.run_train(batch["question"],batch["ques mask"],batch["num pos"],batch["num mask"],\
                            unk,num_start,batch["position"],batch["visible matrix"],\
                            batch["equ len"],batch["equation"],batch["num stack"],CUDA_USE)
        loss = masked_cross_entropy(outputs,
                                    target=batch["equation"],
                                    length=batch["equ len"])
        loss.backward()
        self.optimizer_step()
        batch_loss = loss.item()
        print("[epoch %3d]|[step %4d|%4d] loss[%2.8f]" %
              (self.epoch_i, self.batch_idx, self.batch_nums, batch_loss))
        return batch_loss

    def eval_batch(self, batch):
        unk = self.data_set.vocab.output_dict["UNK"]
        num_start = self.data_set.vocab.num_start
        CUDA_USE = True if torch.cuda.is_available() else False

        test_out=self.run_train(batch["question"],batch["ques mask"],batch["num pos"],batch["num mask"],\
                            unk,num_start,batch["position"],batch["visible matrix"],CUDA_USE)
        batch_val_acc=[]
        batch_equ_acc=[]
        for i in range(len(test_out)):
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                test_out[i], batch["equation"].tolist()[i], self.data_set.vocab,
                batch["num list"][i], batch["num stack"][i])
            batch_val_acc.append(val_ac)
            batch_equ_acc.append(equ_ac)
        return batch_val_acc,batch_equ_acc

    def train_epoch(self):
        self.batch_nums = int(
            self.data_set.train_num / self.args.batch_size) + 1
        print("start training...")
        for epo in range(self.start_epoch, self.args.epochs_num):
            self.scheduler_step()
            self.epoch_i = epo + 1
            epoch_start_time = time.time()
            loss_total = 0.
            # self.eval2train()
            # for batch_idx, batch in enumerate(
            #         self.data_set.load_data(self.args.batch_size, "train")):
            #     self.batch_idx = batch_idx + 1
            #     batch_loss = self.train_batch(batch)
            #     loss_total += batch_loss
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            self.train2eval()
            test_time = time.time()
            for batch in self.data_set.load_data(self.args.batch_size, "test"):
                batch_val_ac, batch_equ_ac = self.eval_batch(batch)
                value_ac += batch_val_ac.count(True)
                equation_ac += batch_equ_ac.count(True)
                eval_total += len(batch_val_ac)
            if value_ac > self.best_value_acc:
                self.save_model(config.MODEL_PATH)
                self.best_value_acc = value_ac
                self.best_equ_acc = equation_ac
                print("---------------------------------------------")
                print("saved model at epoch {}".format(self.epoch_i))
            print("---------------------------------------------")
            print("epoch running time{} test running time{}".format(
                time_since(time.time() - epoch_start_time),
                time_since(time.time() - test_time)))
            print(
                "[epoch %2d|%2d] avr loss[%2.8f] | lr[%1.6f] test equ acc[%2.3f] test ans acc[%2.3f]"
                % (self.epoch_i, self.args.epochs_num, loss_total /
                   self.batch_nums, self.encoder_scheduler.get_lr()[0],
                   equation_ac / eval_total, value_ac / eval_total))
            print("---------------------------------------------")
        print("training finished.")
        print("best value acc:{} equation acc:{}".format(
            self.best_value_acc, self.best_equ_acc))
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
                            current_embeddings_stacks[idx]=b.embedding_stack[idx]
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
                        TreeBeam(torch.tensor(b.score).to(self.device) + tv.squeeze(), current_node_stack,
                                current_embeddings_stacks,
                                current_left_childs, current_out))
            #beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams=beam_sort(current_beams)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                for idx in range(batch_size):
                    if len(b.node_stack[idx]) != 0:
                        flag = False
                        break
                if flag==False:
                    break
            if flag:
                break
        all_out_node=beams[0].out
        all_node_outputs=torch.tensor(all_out_node).to(self.device).long().transpose(0,1)
        return all_node_outputs


if __name__ == "__main__":
    # cuda
    CUDA_USE = True if torch.cuda.is_available() else False
    print("[CUDA USE]:{}".format(CUDA_USE))
    trainer = Trainer()
    trainer.train_epoch()