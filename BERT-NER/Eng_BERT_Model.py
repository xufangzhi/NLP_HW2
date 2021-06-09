from transformers import RobertaForSequenceClassification, AdamW, BertConfig, BertTokenizer, BertPreTrainedModel, BertModel, BertForTokenClassification, RobertaForMultipleChoice, RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random
import sys
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

device = torch.device("cuda:0")

class BERTModel(torch.nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()

        self.roberta = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=9)
        self.labels_num = 9
        self.output_layer = torch.nn.Linear(768, self.labels_num)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        
        out_roberta = self.roberta(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = out_roberta
        return loss, logits


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return pred_flat, labels_flat
    #return np.sum(pred_flat == labels_flat), np.sum(pred_flat != labels_flat)
    
def get_data(path, tokenizer, max_len, labels_map):
    input_ids_list = []
    att_mask_list = []
    labels_list = []
    with open(path, 'r', encoding='utf-8') as file:
        file.readline()
        tokens, labels = [], []
        for line_id, line in enumerate(file):
            tokens, labels = line.strip().split('\t')
            text = ''.join(tokens.split(' '))
            encoded = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True)
            input_ids = encoded['input_ids']
            att_mask = encoded['attention_mask']
            labels = [labels_map[l] for l in labels.split(" ")]
            
            
            if len(labels)<len(input_ids):
                for i in range(0,len(input_ids)-len(labels)):
                    labels.append(labels_map['[PAD]'])
            else:
                labels = labels[:max_len]
                    
            input_ids_list.append(input_ids)
            att_mask_list.append(att_mask)
            labels_list.append(labels)
            
    return input_ids_list, att_mask_list, labels_list


def process_data(raw_data, batch_size, split):
    input_ids_list, att_mask_list, labels_list = raw_data
    inputs = torch.tensor(input_ids_list)
    masks = torch.tensor(att_mask_list)
    

    labels = torch.tensor(labels_list)

    if split=="train":
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        data = TensorDataset(inputs, masks, labels)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader

def evaluate(batch, device, loss, pred, gold, begin_ids, labels_map):
    pred_entities_num = 0
    gold_entities_num = 0
    correct = 0
    
    for j in range(gold.size()[0]):
        for k in range(gold.size()[1]):   
            if gold[j][k].item() in begin_ids:
                gold_entities_num += 1

    for j in range(pred.size()[0]):
        for k in range(pred.size()[1]):
            if pred[j][k].item() in begin_ids and gold[j][k].item() != labels_map["[PAD]"]:
                pred_entities_num += 1

    
    pred_entities_pos = []
    gold_entities_pos = []
    start, end = 0, 0

    for j in range(gold.size()[0]):
        for k in range(gold.size()[1]):
            if gold[j][k].item() in begin_ids:
                start = k
                for i in range(k+1, gold.size()[1]):
                    if gold[j][i].item() == labels_map['[ENT]']:
                        continue
    
                    if gold[j][i].item() == labels_map["[PAD]"] or gold[j][i].item() == labels_map["O"] or gold[j][i].item() in begin_ids:
                        end = i - 1
                        break
                else:
                    end = gold.size()[1] - 1
                gold_entities_pos.append((start, end))

   
    for j in range(pred.size()[0]):
        for k in range(pred.size()[1]):
            if pred[j][k].item() in begin_ids and gold[j][k].item() != labels_map["[PAD]"] and gold[j][k].item() != labels_map["[ENT]"]:
                start = k
                for i in range(k+1, pred.size()[1]): 
                    if gold[j][i].item() == labels_map['[ENT]']:
                        continue
                    if pred[j][i].item() == labels_map["[PAD]"] or pred[j][i].item() == labels_map["O"] or pred[j][i].item() in begin_ids:
                        end = i - 1
                        break
                else:
                    end = pred.size()[1] - 1
                pred_entities_pos.append((start, end))


    for p_entity, g_entity in zip(pred_entities_pos,gold_entities_pos):
        if p_entity == g_entity:
            correct += 1
    #print("gold_entities",gold_entities_pos)
    #print("pred_entities",pred_entities_pos)
    #print("correct:",correct)
    #print("total entity in dataset:",gold_entities_num)
    #print("Report precision, recall, and f1:")
    #print("{:.3f}, {:.3f}, {:.3f}".format(p,r,f1))
    
    correct = 0
    for j in range(gold.size()[0]):
        for k in range(gold.size()[1]):   
            if pred[j][k].item() in begin_ids and pred[j][k].item()==gold[j][k].item():
                correct += 1
                #gold_entities_num += 1

    return pred_entities_num, gold_entities_num, correct


def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, epochs, device, save_model, begin_ids, labels_map):
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        train_loss_list = []
        predict_list = []
        gold_list = []
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0
        correct = 0
        model.train()
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            model.train()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs
            #print(b_labels.size())
            #print(b_labels.contiguous().view(-1).size())
            #print(b_labels)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred, gold = flat_accuracy(logits, label_ids)
            
            
            
            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            predict_list += list(pred)
            gold_list += list(gold)
            #train_acc = accuracy_score(gold_list, predict_list)
            
            #logits = logits.contiguous().view(-1, 9)
            
            #pred_entities, gold_entities, corr = evaluate(batch, device, loss, logits.argmax(dim=-1), b_labels, begin_ids, labels_map)
            #pred_entities_num += pred_entities
            #gold_entities_num += gold_entities
            #correct += corr
            
            
            
            #p = correct/(pred_entities_num+1e-6)
            #r = correct/(gold_entities_num+1e-6)
            #f1 = 2*p*r/(p+r+1e-6)
            f1 = precision_score(gold_list, predict_list, average='weighted')
            pbar.set_description("f1 {0:.4f} loss {1:.4f}".format(f1, train_loss))
            
        if save_model:
            torch.save(model, "checkpoints/BERTModel_"+str(epoch_i+1)+".pth")
            
        dev(model, dev_dataloader, device)
        test(model, test_dataloader, device)

def dev(model, dev_dataloader, device):
    print("")
    print("Running Dev...")

    val_loss_list = []
    predict_list = []
    gold_list = []
    gold_entities_num = 0
    pred_entities_num = 0
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    for batch in tqdm(dev_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        loss,logits = outputs
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred, gold = flat_accuracy(logits, label_ids)
        
        
        val_loss_list.append(loss.item())
        predict_list += list(pred)
        gold_list += list(gold)
        
    val_acc = accuracy_score(gold_list,predict_list)
    val_loss = np.mean(val_loss_list)
    f1 = precision_score(gold_list, predict_list, average='weighted')
    print("f1-score {0:.4f} val_loss {1:.4f}".format(f1, val_loss))



def test(model, test_dataloader, device):
    print("")
    print("Running Test...")

    val_loss_list = []
    predict_list = []
    gold_list = []
    gold_entities_num = 0
    pred_entities_num = 0
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    for batch in tqdm(test_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
        loss,logits = outputs
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred, gold = flat_accuracy(logits, label_ids)

        val_loss_list.append(loss.item())
        predict_list += list(pred)
        gold_list += list(gold)
        
    val_acc = accuracy_score(gold_list,predict_list)
    val_loss = np.mean(val_loss_list)
    f1 = precision_score(gold_list, predict_list, average='weighted')
    print("f1_score {0:.4f} val_loss {1:.4f}".format(f1, val_loss))
    
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--output_model_path", default="./models/tagger_model.bin", type=str, help="Path of the output model.")
    parser.add_argument("--train_path", type=str, required=True, help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True, help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True, help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str, help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=256, help="Batch_size.")
    parser.add_argument("--max_len", type=int, default=32, help="Max of Sequence Length.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning Rate.")
    parser.add_argument("--save", type=bool, default=True, help="Number of epochs.")
    parser.add_argument("--device", type=str, default="gpu", help="Device.")
    args = parser.parse_args()
    
    
    labels_map = {"[PAD]": 0, "[ENT]": 1}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[1].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)

    print("Labels: ", labels_map)
    print("Begin_ids:", begin_ids)
    args.labels_num = len(labels_map)
    
    model = BERTModel()
    print("using BERTModel Now!!!")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')



    if args.device=="gpu":
        device = torch.device("cuda:0")
        model.to(device)
        print(device)
    if args.device=="cpu":
        device = torch.device("cpu") 
        model.cpu()
    
    model.zero_grad()


    batch_size = args.batch_size
    max_len = args.max_len
    lr = args.lr
    epochs = args.epochs
    save_model = args.save

    train_raw_data = get_data(args.train_path, tokenizer, max_len, labels_map)
    dev_raw_data = get_data(args.dev_path, tokenizer, max_len, labels_map)
    test_raw_data = get_data(args.test_path, tokenizer, max_len, labels_map)

    train_dataloader = process_data(train_raw_data, batch_size, "train")
    dev_dataloader = process_data(dev_raw_data, batch_size, "dev")
    test_dataloader = process_data(test_raw_data, batch_size, "test")

    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_raw_data[-1]) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    train(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, epochs, device, save_model, begin_ids, labels_map)
    
if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main()