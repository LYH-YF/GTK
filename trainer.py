import torch
import warnings
warnings.filterwarnings("ignore")
from pre_data import process_math23k
from dataset import DataSet
from models.Model import *
import config
from crossentropyloss import masked_cross_entropy
from eval import compute_prefix_tree_result
# cuda
CUDA_USE = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[CUDA USE]:{}".format(CUDA_USE))

processed_datas, kg, temp_g, copy_nums = process_math23k()
data_set = DataSet(processed_datas, device, random_seed=0)
data_set.build_vocab(kg.segment_vocab, temp_g, copy_nums)
args = config.ARGS
args.input_size = len(temp_g)
args.op_nums = data_set.vocab.op_nums
unk = data_set.vocab.output_dict["UNK"]
num_start = data_set.vocab.num_start
# build model
gtk_model = build_model(config.ARGS, data_set.vocab.input_vocab_len, device)
optimizer = torch.optim.Adam(gtk_model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
batch_nums = int(data_set.train_num / args.batch_size) + 1
best_value_acc=0
best_equ_acc=0
print("start training")
for epo in range(args.epochs_num):
    loss_total = 0.
    batch_idx = 1
    for batch in data_set.load_data(args.batch_size, "train"):

        gtk_model.train()
        '''
        src, mask,num_pos,num_mask,unk,num_start,\
                    pos=None, vm=None,target_length=None,target=None,nums_stack_batch=None,USE_CUDA=False
        '''
        gtk_model.zero_grad()
        outputs=gtk_model(batch["question"],batch["ques mask"],batch["num pos"],batch["num mask"],\
                            unk,num_start,batch["position"],batch["visible matrix"],\
                            batch["equ len"],batch["equation"],batch["num stack"],CUDA_USE)
        loss = masked_cross_entropy(outputs,
                                    target=batch["equation"],
                                    length=batch["equ len"])
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        print("[epoch %3d]|[step %4d|%4d] loss[%2.8f]" %
              ((epo + 1), batch_idx, batch_nums, batch_loss))
        loss_total += batch_loss
        batch_idx += 1
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    gtk_model.eval()
    for batch in data_set.load_data(1, "test"):
        '''
        src, mask,num_pos,num_mask,unk,num_start,\
                pos=None, vm=None,target_length=None,target=None,nums_stack_batch=None,USE_CUDA=False
        '''
        if batch["question"].size(1) != 128:
            continue
        test_out=gtk_model(batch["question"],batch["ques mask"],batch["num pos"],batch["num mask"],\
                            unk,num_start,batch["position"],batch["visible matrix"],CUDA_USE)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(
            test_out, batch["equation"].tolist()[0], data_set.vocab,
            batch["num list"][0], batch["num stack"][0])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1
        print(eval_total)
    if value_ac>best_value_acc:
        checkpoint={"model":gtk_model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "epoch":epo+1}
        torch.save(checkpoint,config.MODEL_PATH.format(epo+1))
        best_value_acc=value_ac
        best_equ_acc=equation_ac
    print("---------------------------------------------")
    print(
        "[epoch %2d|%2d] avr loss[%2.8f] test equ acc[%3.2f] test ans acc[%3.2f]"
        % (epo, args.epochs_num, loss_total / batch_nums,
           equation_ac / eval_total, value_ac / eval_total))
    print("---------------------------------------------")
    print("training finished,best value acc:{} equation acc:{}".format(best_value_acc,best_equ_acc))