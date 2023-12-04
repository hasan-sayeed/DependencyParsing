import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from evaluate import compute_metrics
from torch.utils.data import TensorDataset, DataLoader
from utils import get_word2ix, process_data, parse_file, act_pred

# Setting the seed for reproducibility

torch.manual_seed(42)

# Some hyperparameters
name_glove = '6B'
dim_glove = 50
max_epochs = 20
batch_size = 64
learning_rates = [0.001]
number_of_pos = len(get_word2ix("data/pos_set.txt"))
number_of_classes = len(get_word2ix("data/tagset.txt"))
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Creating the model

class MODEL_mean(nn.Module):
    def __init__(self, pos_vocab_size, pos_embedding_dim, stack_buff_dim, hidden_dim, class_size):
        super(MODEL_mean, self).__init__()
        self.embeddings_pos = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        
        self.linear_stack_buff = nn.Linear(stack_buff_dim, hidden_dim, bias = True)
        self.linear_pos = nn.Linear(pos_embedding_dim, hidden_dim, bias = True)

        self.relu = nn.ReLU()

        self.linear = nn.Linear(hidden_dim, class_size, bias = True)

        

    def forward(self, feature_stack_buff, feature_pos):
        
        embedded_feature_pos = self.embeddings_pos(feature_pos)
        # print("from model before mean")
        # print(embedded_feature_pos.shape)
        embedded_feature_pos = torch.mean(embedded_feature_pos, dim = 1)
        # print("from model")
        # print(embedded_feature_pos.shape)
            
        hidden_stack_buff = self.linear_stack_buff(feature_stack_buff)
        # print(hidden_stack_buff.shape)
        
        
        hidden_pos = self.linear_pos(embedded_feature_pos)

        hidden = self.relu(hidden_stack_buff + hidden_pos)

        return self.linear(hidden)
    
# Reading and preprocessing the data

train_file_path = 'data/tttt.txt'
dev_file_path = 'data/tttt.txt'
test_file_path = 'data/test.txt'

feature_stack_buff_train, feature_pos_train, target_train = process_data(file_path = train_file_path, path_to_pos_set = 'data/pos_set.txt', path_to_tag_set = 'data/tagset.txt', name_glove = name_glove, dim_glove = dim_glove)
feature_stack_buff_dev, feature_pos_dev, target_dev = process_data(file_path = dev_file_path, path_to_pos_set = 'data/pos_set.txt', path_to_tag_set = 'data/tagset.txt', name_glove = name_glove, dim_glove = dim_glove)
feature_stack_buff_test, feature_pos_test, target_test = process_data(file_path = test_file_path, path_to_pos_set = 'data/pos_set.txt', path_to_tag_set = 'data/tagset.txt', name_glove = name_glove, dim_glove = dim_glove)

# Create datasets using the TensorDataset class which takes in parallel tensors.
train_dataset = TensorDataset(
                          torch.tensor(np.array(feature_stack_buff_train), dtype= torch.float),
                          torch.tensor(np.array(feature_pos_train), dtype= torch.float),
                          torch.tensor(np.array(target_train), dtype= torch.float)
                        )
dev_dataset = TensorDataset(
                          torch.tensor(np.array(feature_stack_buff_dev), dtype= torch.float),
                          torch.tensor(np.array(feature_pos_dev), dtype= torch.float),
                          torch.tensor(np.array(target_dev), dtype= torch.float)
                        )
test_dataset = TensorDataset(
                            torch.tensor(np.array(feature_stack_buff_test), dtype= torch.float),
                            torch.tensor(np.array(feature_pos_test), dtype= torch.float),
                            torch.tensor(np.array(target_test), dtype= torch.float)
                        )


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

best_overall_las = 0  # Initialize the best LAS as the worst possible score
best_overall_lr = None
best_overall_epoch = None


for learning_rate in learning_rates:

    best_las = 0  # Initialize the best LAS as the worst possible score
    best_epoch = None

    # Initializing the model
    model = MODEL_mean(pos_vocab_size = number_of_pos, pos_embedding_dim = 50, stack_buff_dim = feature_stack_buff_train.shape[1], hidden_dim = 200, class_size = number_of_classes).to(device)

    # Using cross entropy loss for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # Using Adam optimizer for optimization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []       
        for stack_buff, pos, lab in train_loader:
            model.train()
            optimizer.zero_grad()
            out = model(stack_buff.to(device), pos.to(device, dtype=torch.long)).to(device)     #Forward pass
            loss = loss_fn(out, lab.flatten().to(torch.long).to(device))

            # print(f"Loss: {loss}")   # with shape {loss.shape}

            loss.backward() # computing the gradients
            optimizer.step()  # Performs the optimization

            train_loss.append(loss.item())    # Appending the batch loss to the list

        average_train_loss = np.mean(train_loss)
        print(f"Average training batch loss for Epoch {ep}: {average_train_loss}")

        total_las = 0  # Initialize a variable to accumulate LAS scores
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation during validation
            for stack_buff, pos, lab in dev_loader:
                out = model(stack_buff.to(device), pos.to(device, dtype=torch.long)).to(device)

                words_lists, poss, gold_actions = parse_file(dev_file_path)
                predicted_actions = act_pred(model, file_path=dev_file_path, cwindow=2, name_glove = name_glove, dim_glove = dim_glove, rep_type = 'mean')
                print(gold_actions)
                print(predicted_actions)

                # Compute the LAS score
                _, las = compute_metrics(words_lists, gold_actions, pred_actions = predicted_actions, cwindow=2)
                total_las += las
        
        average_las = total_las / len(dev_loader)  # Compute the average LAS score
        print(f"Average LAS for Epoch {ep}: {average_las}")
        
        # Save the model if it's the best one so far for this learning rate
        if average_las > best_las:  # Notice the '>' because we want to maximize LAS
            best_las = average_las
            best_epoch = ep
            
            model_save_name = f"model_lr={learning_rate}_epoch={ep}.pt"
            torch.save(model.state_dict(), model_save_name)
            
            # Compare with overall best and update if current is better
            if average_las > best_overall_las:  # Notice the '>' because we want to maximize LAS
                best_overall_las = average_las
                best_overall_lr = learning_rate
                best_overall_epoch = ep
                best_model_save_name = model_save_name
    
    # Write best results for this learning rate to a file
    with open("best_results.txt", "a") as f:
        f.write(f"LR={learning_rate}, Epoch={best_epoch}, LAS={best_las}\n")

# Now, you might want to delete the non-best models to save space
for filename in os.listdir():
    if filename.endswith(".pt") and filename != best_model_save_name:
        os.remove(filename)
    else:
        continue

# If you want to save overall best result separately,
# note that the last entry in "best_results.txt" is the overall best.
with open("best_overall_result.txt", "w") as f:
    f.write(f"Best_LR={best_overall_lr}, Best_Epoch={best_overall_epoch}, Best_LAS={best_overall_las}\n")