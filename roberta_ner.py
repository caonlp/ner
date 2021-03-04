import numpy as np
import pandas as pd
import os

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaConfig
from transformers import RobertaForTokenClassification, AdamW
from tqdm import tqdm, trange
from roberta_model import *
from seqeval.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
import codecs

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())


train_sentences_file = codecs.open("wnut_16_data/wnut16_train_sentences.txt", mode ="r", encoding="utf-8")
valid_sentences_file = codecs.open("wnut_16_data/wnut16_valid_sentences.txt", mode ="r", encoding="utf-8")
test_sentences_file = codecs.open("wnut_16_data/wnut16_test_sentences.txt", mode = "r", encoding = "utf-8")


train_labels_file = codecs.open("wnut_16_data/wnut16_train_labels.txt", mode ="r", encoding="utf-8")
valid_labels_file = codecs.open("wnut_16_data/wnut16_valid_labels.txt", mode ="r", encoding="utf-8")
test_labels_file = codecs.open("wnut_16_data/wnut16_test_labels.txt", mode = "r", encoding = "utf-8")


# build train sentences
count_train_sentences = 0
train_sentences = []
for one_line in train_sentences_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).rstrip().split(" ")
    train_sentences.append(one_line)
    count_train_sentences = count_train_sentences + 1

# build valid sentences
count_valid_sentences = 0
valid_sentences = []
for one_line in valid_sentences_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).rstrip().split(" ")
    valid_sentences.append(one_line)
    count_valid_sentences = count_valid_sentences + 1

# build test sentences
count_test_sentences = 0
test_sentences = []
for one_line in test_sentences_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).rstrip().split(" ")
    test_sentences.append(one_line)
    count_test_sentences = count_test_sentences + 1

# build train labels
count_train_labels = 0
train_labels = []
for one_line in train_labels_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).split(" ")
    train_labels.append(list(one_line)[:-1])
    count_train_labels = count_train_labels + 1

# build valid labels
count_valid_labels = 0
valid_labels = []
for one_line in valid_labels_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).split(" ")
    valid_labels.append(list(one_line)[:-1])
    count_valid_labels = count_valid_labels + 1

# build  test labels
count_test_labels = 0
test_labels = []
for one_line in test_labels_file:
    one_line = one_line.strip("\n")
    one_line = str(one_line).split(" ")
    test_labels.append(list(one_line)[:-1])
    count_test_labels = count_test_labels + 1


tags_vals = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
tags_vals.append("PAD")
tag2idx = {t: i for i, t in enumerate(tags_vals)}
print("tags_vals: ", tags_vals)
print("tag2idx: ", tag2idx)


MAX_LEN = 512
bs = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# print("显卡名称: ", torch.cuda.get_device_name(0))

tokenizer = RobertaTokenizer.from_pretrained('roberta_ner_a_model_save', do_lower_case = True)

# special tokens for roberta
special_tokens_dict = {'additional_special_tokens': ["<#>", "<$>"]}

tokenizer.add_special_tokens(special_tokens_dict)


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels_for_train = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(train_sentences, train_labels)]

tokenized_texts_and_labels_for_valid = [
tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(valid_sentences, valid_labels)]

tokenized_texts_and_labels_for_test = [
tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(test_sentences, test_labels)]


train_tokenizer_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_for_train]
train_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_for_train]

valid_tokenizer_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_for_valid]
valid_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_for_valid]

test_tokenizer_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_for_test]
test_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_for_test]

# get train input ids
tr_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenizer_texts], maxlen = MAX_LEN, dtype = "long", value = 0.0, truncating = "post", padding = "post")
# get valid input ids
val_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in valid_tokenizer_texts], maxlen = MAX_LEN, dtype = "long", value = 0.0, truncating = "post", padding = "post")

# get test input ids
te_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenizer_texts], maxlen = MAX_LEN, dtype = "long", value = 0.0, truncating = "post", padding = "post")

# get train tags
tr_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_labels], maxlen = MAX_LEN, value = tag2idx["PAD"], padding = "post", dtype = "long", truncating = "post")
# get valid tags
val_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in valid_labels], maxlen = MAX_LEN, value = tag2idx["PAD"], padding = "post", dtype = "long", truncating = "post")
# get test tags
te_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in test_labels], maxlen = MAX_LEN, value = tag2idx["PAD"], padding = "post", dtype = "long", truncating = "post")


# get train masks
tr_masks = [[float(i > 0) for i in ii] for ii in tr_inputs]
# get valid masks
val_masks = [[float(i > 0) for i in ii] for ii in val_inputs]
# get test masks
te_masks = [[float(i > 0) for i in ii] for ii in te_inputs]

# convert to tensor!
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
te_inputs = torch.tensor(te_inputs)

tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
te_tags = torch.tensor(te_tags)

tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
te_masks = torch.tensor(te_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler = valid_sampler, batch_size = bs)

test_data = TensorDataset(te_inputs, te_masks, te_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = bs)

#config = RobertaConfig.from_pretrained("roberta_ner_a_model_save")
#model = RoBertaTagger(config = config)

model = RobertaForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

# model.resize_token_embeddings(len(tokenizer))
model.to(device)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

epochs = 3
max_grad_norm = 1.0

total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train():

    counter = 1

    # store the average loss after each epoch
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc = "Epoch"):
        # TRAIN loop
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            print(str(counter) + " training......")
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.long()
            b_input_mask = b_input_mask.long()
            b_labels = b_labels.long()

            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()

            # forward pass
            # This will return the loss (rather than the model output)
            outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)

            # get the loss
            loss = outputs[0]

            # backward pass
            loss.backward()

            # track train loss
            total_loss += loss.item()
            print("loss: ", total_loss / counter)

            # gradient clipping
            torch.nn.utils.clip_grad_norm(parameters = model.parameters(), max_norm = max_grad_norm)
            # update parameters
            optimizer.step()
            scheduler.step()
            counter = counter + 1

        # calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_exmaples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.long()
            b_input_mask = b_input_mask.long()
            b_labels = b_labels.long()

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # calculate the accuracy for this batch of valid sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis = 2)])
            true_labels.append(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))




    print("Training Finished!!!")
    torch.save(model, "roberta_ner_model.pkl")


def reload_model():
    trained_model = torch.load("roberta_ner_model.pkl")
    return trained_model


def test():
    model = reload_model().eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.long()
        b_input_mask = b_input_mask.long()
        b_labels = b_labels.long()

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
        # move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # calculate the accuracy for this batch of test sentences
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(test_dataloader)
    print("Test loss: {}".format(eval_loss))
    pred_tags = [tags_vals[p_i] for p, l in zip(predictions, true_labels)
                 for p_i, l_i in zip(p, l) if tags_vals[l_i] != "PAD"]
    test_tags = [tags_vals[l_i] for l in true_labels
                  for l_i in l if tags_vals[l_i] != "PAD"]
    print("Test Accuracy: {}".format(accuracy_score(pred_tags, test_tags)))
    print()


if __name__ == '__main__':
    train()
    #reload_model()
    #test()





