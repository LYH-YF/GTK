import re
from copy import deepcopy
from tools import load_raw_data
from KG import KnowledgeGraph
# def transfer_num(data):  # transfer num into "NUM"
#     print("Transfer numbers...")
#     pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
#     pairs = []
#     generate_nums = []
#     generate_nums_dict = {}
#     copy_nums = 0
#     for d in data:
#         nums = []
#         input_seq = []
#         seg = d["segmented_text"].strip().split(" ")
#         equations = d["equation"][2:]

#         for s in seg:
#             pos = re.search(pattern, s)
#             if pos and pos.start() == 0:
#                 nums.append(s[pos.start(): pos.end()])
#                 input_seq.append("NUM")
#                 if pos.end() < len(s):
#                     input_seq.append(s[pos.end():])
#             else:
#                 input_seq.append(s)
#         if copy_nums < len(nums):
#             copy_nums = len(nums)

#         nums_fraction = []

#         for num in nums:
#             if re.search("\d*\(\d+/\d+\)\d*", num):
#                 nums_fraction.append(num)
#         nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

#         def seg_and_tag(st):  # seg the equation and tag the num
#             res = []
#             for n in nums_fraction:
#                 if n in st:
#                     p_start = st.find(n)
#                     p_end = p_start + len(n)
#                     if p_start > 0:
#                         res += seg_and_tag(st[:p_start])
#                     if nums.count(n) == 1:
#                         res.append("N"+str(nums.index(n)))
#                     else:
#                         res.append(n)
#                     if p_end < len(st):
#                         res += seg_and_tag(st[p_end:])
#                     return res
#             pos_st = re.search("\d+\.\d+%?|\d+%?", st)
#             if pos_st:
#                 p_start = pos_st.start()
#                 p_end = pos_st.end()
#                 if p_start > 0:
#                     res += seg_and_tag(st[:p_start])
#                 st_num = st[p_start:p_end]
#                 if nums.count(st_num) == 1:
#                     res.append("N"+str(nums.index(st_num)))
#                 else:
#                     res.append(st_num)
#                 if p_end < len(st):
#                     res += seg_and_tag(st[p_end:])
#                 return res
#             for ss in st:
#                 res.append(ss)
#             return res

#         out_seq = seg_and_tag(equations)
#         for s in out_seq:  # tag the num which is generated
#             if s[0].isdigit() and s not in generate_nums and s not in nums:
#                 generate_nums.append(s)
#                 generate_nums_dict[s] = 0
#             if s in generate_nums and s not in nums:
#                 generate_nums_dict[s] = generate_nums_dict[s] + 1

#         num_pos = []
#         for i, j in enumerate(input_seq):
#             if j == "NUM":
#                 num_pos.append(i)
#         assert len(nums) == len(num_pos)
#         # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
#         # pairs.append((input_seq, out_seq, nums, num_pos))
#         pairs.append((input_seq, out_seq, nums, num_pos,d["id"]))

#     temp_g = []
#     for g in generate_nums:
#         if generate_nums_dict[g] >= 5:
#             temp_g.append(g)
#     return pairs, temp_g, copy_nums


def transfer_num_2(data):  # transfer num into "NUM"
    '''
    {"question":sent_know,"position":position,"visible matrix":vm,"equation":data["equation"],
                    "num list":data["num list"],"id":data["id"]}
    '''
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    #pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    new_datas = []
    for d in data:
        nums = []
        input_seq = []
        seg = d["question"]
        equations = d["equation"][2:]

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
        nums_fraction = sorted(nums_fraction,
                               key=lambda x: len(x),
                               reverse=True)

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
            "num pos": num_pos,
            "visible matrix": d["visible matrix"],
            "position": d["position"],
            "id": d["id"],
            "sent len": d["sent len"],
            "ans": d["ans"]
        })

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 5:
            temp_g.append(g)
    return new_datas, temp_g, copy_nums


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [
                    ")", "]"
            ] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def prefix(datas):
    '''
    {"question":input_seq,"equation":out_seq,"num list":nums,"num pos":num_pos,
                            "visible matrix":d["visible matrix"],"position":d["position"],
                            "id":d["id"],"sent len":d["sent len"],"ans":d["ans"]}
    '''
    new_datas = []
    for data in datas:
        new_data = data
        new_data["equation"] = from_infix_to_prefix(data["equation"])
        new_datas.append(new_data)
    return new_datas


def pair2dict(pairs):
    datas = []
    for pair in pairs:
        sentence, equation, num_list, num_pos, ques_id = pair[0], pair[
            1], pair[2], pair[3], pair[4]
        datas.append({
            "question": sentence,
            "equation": equation,
            "num list": num_list,
            "num pos": num_pos,
            "id": ques_id
        })
    return datas


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [
                    ")", "]"
            ] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def process_math23k():
    json_datas = load_raw_data("data/Math_23K.json")
    #json_datas = json_datas[:1000]

    #pairs, temp_g, copy_nums=transfer_num(json_datas)
    #json_datas=pair2dict(pairs)
    kg = KnowledgeGraph("data/kg_cleared.json")
    processed_datas = []
    # print("add knowledge/transfer nums...")
    # know_sent_batch, position_batch, visible_matrix_batch,\
    #     sent_len_batch,num_pos_batch,num_list_batch,\
    #         ans_batch,id_batch,equation_batch,temp_g,copy_nums,seg_batch=kg.add_knowledge_with_vm(json_datas,max_entities=2)
    # for know_sent, position, visible_matrix,sent_len,num_pos,num_list,ans,id_,equation in zip(know_sent_batch, position_batch, visible_matrix_batch,sent_len_batch,num_pos_batch,num_list_batch,ans_batch,id_batch,equation_batch):
    #     data={"question":know_sent,"position":position,"visible matrix":visible_matrix,
    #             "sent len":sent_len,"num pos":num_pos,"num list":num_list,
    #             "ans":ans,"id":id_,"equation":equation}
    #     if len(position)!=128:
    #         print(1)
    #     processed_datas.append(data)
    #                     "equation":data["equation"],"id":data["id"],"sent len":sent_len[0],"ans":data["ans"]}
    for i, data in enumerate(json_datas):
        sent_know, position, vm, sent_len, _ = kg.add_knowledge_with_vm(
            [data], max_entities=2)
        processed_data = {
            "question": sent_know[0],
            "position": position[0],
            "visible matrix": vm[0],
            "equation": data["equation"],
            "id": data["id"],
            "sent len": sent_len[0],
            "ans": data["ans"]
        }
        processed_datas.append(processed_data)
        print("\radd knowledge to sentence %5d" % i, end="")
    print()
    processed_datas, temp_g, copy_nums = transfer_num_2(processed_datas)
    processed_datas = prefix(processed_datas)
    return processed_datas, kg, temp_g, copy_nums


if __name__ == "__main__":
    pass
    # processed_datas,kg,temp_g,copy_nums=process_math23k()
    # data_set=DataSet(processed_datas,random_seed=0)
    # data_set.build_vocab(kg.segment_vocab,temp_g,copy_nums)
    # for batch in data_set.load_data(32,"train"):
    #     pass
    # data=[{
    #     "segmented_text":"有 一段 绳子 长 5 米，4 根 这样 的 绳子 一共 多少 米 ？",
    #     "equation":"x=5*4",
    #     "id":3
    # },
    # {
    #     "segmented_text":"有 一段 绳子 长 5 米，4 根 这样 的 绳子 一共 多少 米 ？",
    #     "equation":"x=5*4",
    #     "id":3
    # },
    # {
    #     "segmented_text":"有 一段 绳子 长 5 米，6 根 这样 的 绳子 一共 多少 米 ？",
    #     "equation":"x=5*4",
    #     "id":3
    # }
    # ]
    # pairs,x,y=transfer_num(data)
    # print(pairs)
    # print(x)
    # print(y)
    # sentence,equation,num_list,copy_numlist=pairs[0][0],pairs[0][1],pairs[0][2],pairs[0][4]
    # print(sentence)