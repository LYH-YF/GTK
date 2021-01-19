import random
import re
import torch
import config

def get_num_mask(num_size_batch, generate_nums):
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    return num_mask


def get_seq_mask(input_length_batch, max_seq_len):
    seq_mask = []
    #max_len = max(input_length_batch)
    for i in input_length_batch:
        seq_mask.append([1 for _ in range(i)] +
                        [0 for _ in range(i, max_seq_len)])
    return seq_mask


# def get_seq_mask(input_length_batch):
#     seq_mask = []
#     max_len = max(input_length_batch)
#     for i in input_length_batch:
#         seq_mask.append([1 for _ in range(i)] + [0 for _ in range(i, max_len)])
#     return seq_mask
class Vocab(object):
    def __init__(self, kg_vocab, generate_num, copy_nums):
        super().__init__()
        self.word2index = {"PAD": 0, "NUM": 1, "UNK": 2}
        self.index2word = ["PAD", "NUM", "UNK"]
        self.output = []
        self.output_dict = {}
        self.word2count = {}
        self.input_vocab_len = len(self.index2word)
        self.output_len = 0
        self.op_list = ["PAD", "+", "-", "*", "/", "^"]
        self.op_nums = len(self.op_list)
        self.generate_num = generate_num
        self.add_kg_vocab(kg_vocab)
        self.build_output_vocab(generate_num, copy_nums)

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            if word == 0:
                break
            else:
                if re.search("N\d+|NUM|\d+", word):
                    continue
                if word not in self.index2word:
                    self.word2index[word] = self.input_vocab_len
                    self.input_vocab_len += 1
                    self.word2count[word] = 1
                    self.index2word.append(word)
                else:
                    self.word2count[word] += 1

    def add_kg_vocab(self, kg_vocab):
        for word in kg_vocab:
            if word not in self.index2word:
                self.word2index[word] = self.input_vocab_len
                self.input_vocab_len += 1
                self.word2count[word] = 1
                self.index2word.append(word)
            else:
                self.word2count[word] += 1

    def build_output_vocab(self, generate_num, copy_nums):
        self.output = self.op_list+ generate_num + ["N" + str(i) for i in range(copy_nums)] +\
                          ["UNK"]
        self.num_start = len(self.op_list)
        for j in self.output:
            self.output_dict[j] = self.output_len
            self.output_len += 1


class DataSet(object):
    def __init__(self, datas, device, random_seed=None):
        super().__init__()
        self.random_seed = random_seed
        self.trainset, self.testset = self.read_dataset(datas)
        self.train_num = len(self.trainset)
        self.test_num = len(self.testset)
        self.device = device

    def build_vocab(self, kg_vocab, generate_num, copy_nums):
        vocab = Vocab(kg_vocab, generate_num, copy_nums)
        for data in self.trainset:
            vocab.add_sen_to_vocab(data["question"])
        self.vocab = vocab

    def read_dataset(self, datas):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        random.shuffle(datas)
        length = len(datas)
        train = datas[int(length / 5):]
        test = datas[:int(length / 5)]
        return train, test

    def load_data(self, batchsize, type):
        if type == "train":
            datas = self.trainset
            num_total = self.train_num
        else:
            datas = self.testset
            num_total = self.test_num
        batch_num = int(num_total / batchsize) + 1
        for batch_i in range(batch_num):
            start_idx = batch_i * batchsize
            end_idx = (batch_i + 1) * batchsize
            if end_idx <= num_total:
                batch_data = datas[start_idx:end_idx]
            else:
                batch_data = datas[start_idx:num_total]
            if batch_data != []:
                if config.ARGS.encoder=="rnn":
                    batch_data=self.batch2tensor_for_rnn(batch_data)
                else:
                    batch_data = self.batch2tensor(batch_data)
                yield batch_data

    def batch2tensor(self, batch_data):
        '''
        {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos,
                            "visible matrix":d["visible matrix"],"position":d["position"],"id":d["id"]}
        '''
        ques_tensor_batch = []
        equ_tensor_batch = []
        vm_batch = []
        num_list_batch = []
        num_pos_batch = []
        position_batch = []
        id_batch = []
        #num_mask_batch=[]
        ques_mask_batch = []
        equ_len_batch = []
        ques_len_batch = []
        num_size_batch = []
        num_mask_batch = []
        ques_mask_batch = []
        ans_batch = []
        num_stack_batch = []
        for data in batch_data:
            ques_tensor = []
            equ_tensor = []
            sentence = data["question"]
            equation = data["equation"]
            vm_batch.append(data["visible matrix"])
            num_list_batch.append(data["num list"])
            num_pos_batch.append(data["num pos"])
            position_batch.append(data["position"])
            id_batch.append(data["id"])
            ques_len_batch.append(data["sent len"])
            ans_batch.append(data["ans"])
            num_size_batch = [len(num_pos) for num_pos in num_pos_batch]
            num_stack_batch.append(
                self.build_num_stack(equation, data["num list"]))
            for word in sentence:
                if word == 0:
                    idx = self.vocab.word2index["PAD"]
                else:
                    try:
                        idx = self.vocab.word2index[word]
                    except:
                        idx = self.vocab.word2index["UNK"]
                ques_tensor.append(idx)
            if len(ques_tensor) > 128:
                print(1)
            for word in equation:
                try:
                    idx = self.vocab.output_dict[word]
                except:
                    idx = self.vocab.output_dict["UNK"]
                equ_tensor.append(idx)
            equ_len_batch.append(len(equ_tensor))
            ques_tensor_batch.append(ques_tensor)
            equ_tensor_batch.append(equ_tensor)
        max_equ_len = max(equ_len_batch)
        #num_mask_batch=[[1]*l+[0]*(max_equ_len-l) for l in equ_len_batch]
        for i, l in enumerate(equ_len_batch):
            equ_tensor_batch[i] += [self.vocab.output_dict["PAD"]
                                    ] * (max_equ_len - l)
        #ques_mask_batch=[[1]*l+[0]*(len(ques_tensor_batch[i])-l) for i,l in enumerate(ques_len_batch)]
        ques_mask_batch = get_seq_mask(ques_len_batch,
                                       len(ques_tensor_batch[0]))
        num_mask_batch = get_num_mask(num_size_batch, self.vocab.generate_num)
        # to tensor
        ques_tensor_batch = torch.tensor(ques_tensor_batch).to(self.device)
        equ_tensor_batch = torch.tensor(equ_tensor_batch).to(self.device)
        vm_batch = torch.tensor(vm_batch).to(self.device)
        position_batch = torch.tensor(position_batch).to(self.device)
        ques_mask_batch = torch.tensor(ques_mask_batch).to(self.device)
        num_mask_batch = torch.tensor(num_mask_batch).to(self.device)
        ques_len_batch=torch.tensor(ques_len_batch).to(self.device).long()
        return {
            "question": ques_tensor_batch,
            "equation": equ_tensor_batch,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "visible matrix": vm_batch,
            "position": position_batch,
            "id": id_batch,
            "num mask": num_mask_batch,
            "ques mask": ques_mask_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "ques len":ques_len_batch,
        }
    def batch2tensor_for_rnn(self,batch_data):
        batch_data=sorted(batch_data,key=lambda x:x["sent len"],reverse=True)
        ques_tensor_batch = []
        equ_tensor_batch = []
        vm_batch = []
        num_list_batch = []
        num_pos_batch = []
        position_batch = []
        id_batch = []
        #num_mask_batch=[]
        ques_mask_batch = []
        equ_len_batch = []
        ques_len_batch = []
        num_size_batch = []
        num_mask_batch = []
        ques_mask_batch = []
        ans_batch = []
        num_stack_batch = []
        for data in batch_data:
            ques_tensor = []
            equ_tensor = []
            sentence = data["question"]
            equation = data["equation"]
            vm_batch.append(data["visible matrix"])
            num_list_batch.append(data["num list"])
            num_pos_batch.append(data["num pos"])
            position_batch.append(data["position"])
            id_batch.append(data["id"])
            ques_len_batch.append(data["sent len"])
            ans_batch.append(data["ans"])
            num_size_batch = [len(num_pos) for num_pos in num_pos_batch]
            num_stack_batch.append(
                self.build_num_stack(equation, data["num list"]))
            for word in sentence:
                if word == 0:
                    idx = self.vocab.word2index["PAD"]
                else:
                    try:
                        idx = self.vocab.word2index[word]
                    except:
                        idx = self.vocab.word2index["UNK"]
                ques_tensor.append(idx)
            for word in equation:
                try:
                    idx = self.vocab.output_dict[word]
                except:
                    idx = self.vocab.output_dict["UNK"]
                equ_tensor.append(idx)
            equ_len_batch.append(len(equ_tensor))
            ques_tensor_batch.append(ques_tensor)
            equ_tensor_batch.append(equ_tensor)
        max_ques_len=max(ques_len_batch)
        for i,l in enumerate(ques_len_batch):
            ques_tensor_batch[i]+=[self.vocab.word2index["PAD"]]*(max_ques_len-l)
        max_equ_len = max(equ_len_batch)
        #num_mask_batch=[[1]*l+[0]*(max_equ_len-l) for l in equ_len_batch]
        for i, l in enumerate(equ_len_batch):
            equ_tensor_batch[i] += [self.vocab.output_dict["PAD"]
                                    ] * (max_equ_len - l)
        #ques_mask_batch=[[1]*l+[0]*(len(ques_tensor_batch[i])-l) for i,l in enumerate(ques_len_batch)]
        ques_mask_batch = get_seq_mask(ques_len_batch,
                                       max_ques_len)
        num_mask_batch = get_num_mask(num_size_batch, self.vocab.generate_num)
        # to tensor
        ques_tensor_batch = torch.tensor(ques_tensor_batch).to(self.device)
        equ_tensor_batch = torch.tensor(equ_tensor_batch).to(self.device)
        vm_batch = torch.tensor(vm_batch).to(self.device)
        position_batch = torch.tensor(position_batch).to(self.device)
        ques_mask_batch = torch.tensor(ques_mask_batch).to(self.device)
        num_mask_batch = torch.tensor(num_mask_batch).to(self.device)
        ques_len_batch=torch.tensor(ques_len_batch).to(self.device).long()
        return {
            "question": ques_tensor_batch,
            "equation": equ_tensor_batch,
            "equ len": equ_len_batch,
            "num list": num_list_batch,
            "num pos": num_pos_batch,
            "visible matrix": vm_batch,
            "position": position_batch,
            "id": id_batch,
            "num mask": num_mask_batch,
            "ques mask": ques_mask_batch,
            "num stack": num_stack_batch,
            "ans": ans_batch,
            "ques len":ques_len_batch,
        }
    def build_num_stack(self, equation, num_list):
        num_stack = []
        for word in equation:
            temp_num = []
            flag_not = True
            if word not in self.vocab.output:
                flag_not = False
                for i, j in enumerate(num_list):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(num_list))])
        num_stack.reverse()
        return num_stack