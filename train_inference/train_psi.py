
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

    

class MyDataSet(Dataset):
    def __init__(self):
        file_name = './train_inference/train_data'
        num_data = 15000
        self.samples = []
        self.output_labels = []

        
        for i in range(num_data):
            attentive_data = np.load(file_name+'/attentive'+str(i)+'.npz',allow_pickle=True)
            states = attentive_data['traj']
            u_h = attentive_data['human_input']
            beta = attentive_data['t_beta']
            
            sample, label = self.states_to_encoder_input_torch(states, u_h, 1.0)
            self.samples.append(sample)
            self.output_labels.append(label)
        
        for i in range(num_data,2*num_data):
            attentive_data = np.load(file_name+'/distracted'+str(i)+'.npz',allow_pickle=True)
            states = attentive_data['traj']
            u_h = attentive_data['human_input']
            beta = attentive_data['t_beta']
            
            sample, label = self.states_to_encoder_input_torch(states, u_h, 0.0)
            self.samples.append(sample)
            self.output_labels.append(label)
            
        

    def gen_dataset(self):

        
        inputs = torch.stack(self.samples).to(torch.device("cuda"))  
        labels = torch.stack(self.output_labels).to(torch.device("cuda"))

        input_size = len(inputs)
        perm = torch.randperm(input_size)
        inputs = inputs[perm]
        labels = labels[perm]
        dataset =  torch.utils.data.TensorDataset(inputs,labels)
        train_size = int(0.8 * input_size)
        val_size = int(0.1 * input_size)
        test_size = input_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        return train_dataset, val_dataset, test_dataset

          
    def states_to_encoder_input_torch(self, states, u_h, theta):
       
        time_horizon = len(states)
        input_data_arr = torch.zeros(time_horizon, 10) #input_dim = 10
        for i in range(time_horizon):
            input_data_arr[i,:] = torch.tensor([ states[i][0],  #human x            
                            states[i][1], #human y    
                            states[i][2], #human psi    
                            states[i][3], #human vel  
                            states[i][4], #ego x            
                            states[i][5], #ego y    
                            states[i][6], #ego psi    
                            states[i][7], #ego vel                       
                            u_h[i][0], #human acceleration
                            u_h[i][1]]) #human steering angle
        
        output_label_arr = torch.tensor([theta for _ in range(time_horizon)])
        
        return input_data_arr, output_label_arr

class Encoder(nn.Module):
    def __init__(self,input_size = 10, hidden_size = 5, num_layers = 2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
    def forward(self, x):        
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class EncoderClassifier(nn.Module):
    def __init__(self, args, drop_out = 0):
        super(EncoderClassifier, self).__init__()
        self.device = args['device']

        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.output_size = args['latent_size']
        self.seq_len = args['seq_len']
        self.num_layers = 2

        self.dropout = nn.Dropout(drop_out)
        self.lstm_enc = Encoder(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
    
    def forward(self, x, intention_label=None):
        
        batch_size, seq_len, input_dim = x.shape
        if seq_len != self.seq_len:
            print("Warning !! sequence lentgh is not matched")
            return

        # Forward pass through the encoder
        enc_hidden = self.lstm_enc(x)
        enc_h = enc_hidden[0][-1,:,:].view(batch_size, self.hidden_size).to(self.device)

        # Forward pass through the classifier
        enc_h = self.dropout(enc_h)
        intention_logits = self.fc(enc_h)
        

        if intention_label is None:
            intention_binary_value = (intention_logits >= 0.5).to(torch.float32)
            intention_prob = intention_logits
            return intention_binary_value, intention_logits
        else:
            intention_label = intention_label.to(torch.float32)
            # During training, compute binary cross-entropy loss and MSE loss
            binary_loss = self.criterion(intention_logits, intention_label.reshape(len(intention_label),1))
            correct = torch.sum((intention_logits >= 0.5).to(torch.float32)==intention_label.reshape(len(intention_label),1))
            correct_ = correct.item() / len(intention_label)
            return binary_loss, correct_, intention_logits


class DrivingStyleClassifier():
    def __init__(self, args, model_load = False, model_id = 100):

        self.train_data = None
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.model_id = model_id 

        ## args
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.output_size = args['latent_size']
        self.seq_len = args['seq_len']
        self.args = args

        self.file_dir = './train_inference/encoder/'
        if model_load:
            self.model_load(model_id)
    
    def set_train_loader(self,data_loader):
        self.train_loader = data_loader

    def set_test_loader(self,data_loader):
        self.test_loader = data_loader
    
    def reset_args(self,args):
        self.args = args
        self.input_size = args['input_size']
        self.hidden_size = args['hidden_size']
        self.output_size = args['latent_size']
        self.seq_len = args['seq_len']

    def model_load(self, model_id =None):
        if model_id is None:
            model_id = self.model_id
        saved_data = torch.load(self.file_dir+f"cont_encoder_{model_id}.model")            
        loaded_args= saved_data['train_args']
        self.reset_args(loaded_args)

        model_state_dict = saved_data['model_state_dict']
        self.model = EncoderClassifier(self.args).to(device='cuda')                
        self.model.to(torch.device("cuda"))
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
    
    def model_save(self, model_id= None):
        if model_id is None:
            model_id = self.model_id
        save_dir = self.file_dir +f"cont_encoder_{model_id}.model" 
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'train_args': self.args
            }, save_dir)
            
        # torch.save(self.model.state_dict(), save_dir )
        print("model has been saved in "+ save_dir)

    def get_drivingstyle(self, data):
        with torch.no_grad():
            intention, intention_prob = self.model(data)
        if len(intention) == 1:
            return float(intention.item()), float(intention_prob.item())
        else:
            return intention, intention_prob
        
    def train(self):
        if self.train_loader is None:
            print("Warning !! there is no training data")
            return
        if self.model is None:
            model = EncoderClassifier(self.args, drop_out=0.5).to(device='cuda')
        else:
            model = self.model.to(device='cuda')
        

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args['learning_rate'])

        ## interation setup
        epochs = tqdm(range(self.args['max_iter'] // len(self.train_loader) + 1))

        ## training
        count = 0
       
        eval_losses_arr = []
        train_losses_arr = []
        eval_losses = 0
        train_losses = 0

         
        corrects = 0
        corrects_arr = []
        corrects_eval = 0
        corrects_eval_arr = []
        for epoch in epochs:
            model.train()
            optimizer.zero_grad()
            train_iterator = tqdm(
                enumerate(self.train_loader), total=len(self.train_loader), desc="training")
            
            train_losses_itr = 0
            accuracy_itr = 0

            for i, batch_data in train_iterator:

                if count > self.args['max_iter']:
                    print("count exceed")
                    self.model = model
                    self.model_save(model_id=epoch) 

                    return model
                count += 1

                train_data = batch_data[0].to(self.args['device'])
                intention_label = batch_data[1].to(self.args['device'])

                num_samples, num_time_steps, input_dim = train_data.shape
                num_sequences = num_time_steps - self.seq_len + 1
                loss_total = 0
                correct_total = 0

                for j in range(num_sequences):
                    sequence = train_data[:, j:j+self.seq_len, :]
                    label = intention_label[:, j+self.seq_len-1]

                    loss, correct, _ = model(sequence, intention_label=label)

                    # Backpropagation and optimization
                    loss_total += loss 
                    correct_total += correct
                    loss.backward()
                    optimizer.step()

                loss_ = loss_total.item()/num_sequences
                correct_ = correct_total/num_sequences
                train_iterator.set_postfix({"loss": loss_})  # Display loss in the progress bar
            
                train_losses_itr += loss_ 
                accuracy_itr += correct_

            train_losses = train_losses_itr/len(train_iterator) 
            corrects = accuracy_itr/len(train_iterator)
            


            model.eval()
            eval_loss = 0
            test_iterator = tqdm(
                enumerate(self.test_loader), total=len(self.test_loader), desc="testing"
            )
            eval_losses_itr = 0
            accuracy_eval_itr = 0 
            with torch.no_grad():
                for i, batch_data in test_iterator:
                    train_data = batch_data[0].to(self.args['device'])
                    intention_label = batch_data[1].to(self.args['device'])

                    num_samples, num_time_steps, input_dim = train_data.shape
                    num_sequences = num_time_steps - self.seq_len + 1
                    total_loss = 0
                    total_correct = 0
                    for j in range(num_sequences):
                        sequence = train_data[:, j:j+self.seq_len, :]
                        label = intention_label[:, j+self.seq_len-1]

                        loss, correct, _ = model(sequence, intention_label=label)

                        total_loss += loss
                        total_correct += correct
                    
                    loss_ = total_loss.item()/num_sequences
                    correct_ = total_correct/num_sequences
                    test_iterator.set_postfix({"eval_loss": float(loss_)})
                
                    eval_losses_itr += loss_
                    accuracy_eval_itr += correct_
    
            eval_losses = eval_losses_itr/len(test_iterator)
            corrects_eval = accuracy_eval_itr/len(test_iterator)
            print("Evaluation Score : [{}]".format(float(loss_)))

            self.model = model
                
            # if epoch >0 and epoch%10 == 0:
            #     eval_losses_arr.append(eval_losses)
            #     train_losses_arr.append(train_losses)
            #     corrects_arr.append(corrects)
            #     corrects_eval_arr.append(corrects_eval)
            #     eval_losses = 0
            #     train_losses = 0
            #     corrects = 0
            #     corrects_eval = 0
                
            #     plt.figure(1)
            #     plt.plot(eval_losses_arr,'r')
            #     plt.plot(train_losses_arr,'g')
                
            #     plt.figure(2)
            #     plt.plot(corrects_arr,'b')
            #     plt.plot(corrects_eval_arr,'r')
            #     plt.pause(0.001)
            #     plt.draw()


            if epoch%200 == 0:
                self.model_save(model_id=epoch) 
        
        # self.model_save(model_id=epoch) 

if __name__=="__main__":
    data = MyDataSet()
    train_dataset, val_dataset, test_dataset = data.gen_dataset()

    args_ = {
                "batch_size": 512,
                "device": torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
                "input_size": 10,
                "output_size": 1,
                "hidden_size": 64,
                "latent_size": 1,
                "learning_rate": 0.00001, 
                "max_iter": 40000,
                "seq_len" : 5
            }
    batch_size = args_["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    internalstate_encoder = DrivingStyleClassifier(args = args_)
    internalstate_encoder.set_train_loader(train_loader)
    internalstate_encoder.set_test_loader(test_loader)

    internalstate_encoder.train()
    plt.show()

