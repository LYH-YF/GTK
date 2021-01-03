import numpy as np
import re
import jieba
import config
from tools import load_json_data


def transfer_num_2(sentence, equation, temp_g,
                   copy_nums):  # transfer num into "NUM"
    '''
    
    '''
    #print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    #pairs = []
    generate_nums = []
    generate_nums_dict = {}
    new_datas = []
    nums = []
    input_seq = []
    seg = sentence
    equations = equation[2:]

    for s in seg:
        if s == 0:
            input_seq.append(s)
        else:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start():pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end():])
            else:
                input_seq.append(s)
    if copy_nums < len(nums):
        copy_nums = len(nums)

    nums_fraction = []

    for num in nums:
        if re.search("\d*\(\d+/\d+\)\d*", num):
            nums_fraction.append(num)
    nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

    def seg_and_tag(st):  # seg the equation and tag the num
        res = []
        for n in nums_fraction:
            if n in st:
                p_start = st.find(n)
                p_end = p_start + len(n)
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                if nums.count(n) == 1:
                    res.append("N" + str(nums.index(n)))
                else:
                    res.append(n)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
        pos_st = re.search("\d+\.\d+%?|\d+%?", st)
        if pos_st:
            p_start = pos_st.start()
            p_end = pos_st.end()
            if p_start > 0:
                res += seg_and_tag(st[:p_start])
            st_num = st[p_start:p_end]
            if nums.count(st_num) == 1:
                res.append("N" + str(nums.index(st_num)))
            else:
                res.append(st_num)
            if p_end < len(st):
                res += seg_and_tag(st[p_end:])
            return res
        for ss in st:
            res.append(ss)
        return res

    out_seq = seg_and_tag(equations)
    for s in out_seq:  # tag the num which is generated
        if s[0].isdigit() and s not in generate_nums and s not in nums:
            generate_nums.append(s)
            generate_nums_dict[s] = 0
        if s in generate_nums and s not in nums:
            generate_nums_dict[s] = generate_nums_dict[s] + 1

    num_pos = []
    for i, j in enumerate(input_seq):
        if j == "NUM":
            num_pos.append(i)
    assert len(nums) == len(num_pos)

    new_datas.append({
        "question": input_seq,
        "equation": out_seq,
        "num list": nums,
        "num pos": num_pos
    })

    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return new_datas, temp_g, copy_nums


def split_number(text_list):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    new_text = []
    for s in text_list:
        pos = re.search(pattern, s)
        if pos and pos.start() == 0:
            num = s[pos.start():pos.end()]
            new_text.append(num)
            if pos.end() < len(s):
                new_text.append(s[pos.end():])
        else:
            new_text.append(s)
    return new_text


def joint_number(text_list):
    new_list = []
    i = 0
    while i < len(text_list):
        if text_list[i] == '(' and i + 4 < len(text_list) and text_list[
                i + 4] == ')':
            sub = ''.join(text_list[i:i + 5])
            new_list.append(sub)
            i = i + 5
        else:
            new_list.append(text_list[i])
            i += 1
    return new_list


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """
    def __init__(self, kg_path, predicate=False):
        self.predicate = predicate
        #self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.kg_file_path = kg_path
        self.lookup_table, self.segment_vocab = self._create_lookup_table()
        #self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        #self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        lookup_table = {}
        vocab_list = []
        kg_list = load_json_data(self.kg_file_path)
        for kg in kg_list:
            subj = kg[0]
            attr = kg[1]
            obj = kg[2]
            attr = [x for x in jieba.cut(attr)]
            attr = joint_number(attr)
            attr = split_number(attr)
            obj = [x for x in jieba.cut(obj)]
            obj = joint_number(obj)
            obj = split_number(obj)
            try:
                lookup_table[subj]
                lookup_table[subj].append(attr + obj)
            except:
                lookup_table[subj] = [attr + obj]

            vocabs = [subj] + attr + obj
            for vocab in vocabs:
                if vocab not in vocab_list:
                    vocab_list.append(vocab)

        return lookup_table, vocab_list

    def add_knowledge_with_vm(self,
                              datas,
                              max_entities,
                              add_pad=True,
                              max_length=128):
        """
        input: datas - list of dict data, e.g., [{"question":"sentence","eqution":"5*4"...}]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        #split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        #split_sent_batch = sent_batch.split(" ")
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        num_pos_batch = []
        num_list_batch = []
        ans_batch = []
        id_batch = []
        equation_batch = []
        sent_len_batch = []
        temp_g = []
        copy_nums = 0
        for data in datas:
            ans_batch.append(data["ans"])
            id_batch.append(data["id"])
            # if "22m" in data["original_text"]:
            #     print(1)
            #equation = data["equation"]
            #split_sent=data["segmented_text"].strip().split(" ")
            #split_sent2=" ".join(jieba.cut(data["original_text"])).split(" ")
            split_sent2 = [word for word in jieba.cut(data["original_text"])]
            split_sent = joint_number(split_sent2)
            split_sent = split_number(split_sent)
            sent_len = 0
            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []
            for token in split_sent:
                sent_len += 1
                #entities = list(self.lookup_table.get(token, []))[:max_entities]
                entities = self.lookup_table.get(token, [])[:max_entities]
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    #token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    #token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                abs_idx = token_abs_idx[-1]

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities:
                    sent_len += 1
                    ent_pos_idx = [
                        token_pos_idx[-1] + i for i in range(1,
                                                             len(ent) + 1)
                    ]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            sent_len_batch.append(sent_len)
            #     if "NUM" in token:
            #         num_pos.append(token_pos_idx)
            # num_pos_batch.append(num_pos)

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word]
                    seg += [0]
                else:
                    #add_word = list(word)
                    #know_sent=add_word
                    know_sent += [word]
                    seg += [0]
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # data_,temp_g,copy_nums=transfer_num_2(know_sent,equation,temp_g,copy_nums)
            # data_=data_[0]
            # '''
            # {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos}
            # '''
            # know_sent=data_["question"]
            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [
                        idx for ent in item[1] for idx in ent
                    ]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if (src_length > 128):
                print(1)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix,
                                        ((0, pad_num), (0, pad_num)),
                                        'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            # num_pos_batch.append(data_["num pos"])
            # equation_batch.append(data_["equation"])
            # num_list_batch.append(data_["num list"])
            # '''
            # {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos}
            # '''
            # seg_batch.append(seg)

        #return know_sent_batch, position_batch, visible_matrix_batch,sent_len_batch,num_pos_batch,num_list_batch,ans_batch,id_batch,equation_batch,temp_g,copy_nums,seg_batch
        return know_sent_batch, position_batch, visible_matrix_batch, sent_len_batch, seg_batch


if __name__ == "__main__":
    x = jieba.cut_for_search("表达式模型")
    for a in x:
        print(a)
    x = jieba.cut("表达式模型")
    for a in x:
        print(a)