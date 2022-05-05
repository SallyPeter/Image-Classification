from Util import *

# User Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--save_dir')
parser.add_argument('--architecture')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_layers')
parser.add_argument('--epochs')
parser.add_argument('--device')
parser.add_argument('--batch_size')
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
architecture = args.architecture
learning_rate = args.learning_rate
hidden_layers = args.hidden_layers
epochs = args.epochs
device = args.device
batch_size = args.batch_size

# When no input from user
device = "cpu" if device is None else device
learning_rate = 0.001 if learning_rate is None else float(learning_rate)
hidden_layers = 4096 if hidden_layers is None else int(hidden_layers)
batch_size = 32 if batch_size is None else int(batch_size)
epochs = 25 if epochs is None else int(epochs)
save_dir = 'checkpoint.pth' if save_dir is None else save_dir
input_, output_ = 25088, 102 if architecture == "vgg13" or architecture == "vgg19_bn" else print("what model do you want to use")

if(data_dir == None) or (save_dir == None) or (architecture == None) or (learning_rate == None) or (hidden_layers == None) or (epochs == None) or (device == None):
    print("One or more parameters are none")
    exit()

    

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True) 
               for x in ['train_set', 'valid_set', 'test_set']}

model = models.vgg13(pretrained=True) if architecture == "vgg13" else models.vgg19_bn(pretrained=True) if architecture == "vgg19_bn" else print("Model not available")


# Freeze parameters so we don't backprop through them
for param in model.features.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(input_, hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_layers, output_),
                                 nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#model.cuda();
model.to(device)
print(model)

# Training the model

for e in range(1, epochs+1):

    for dataset in ['train_set', 'valid_set']:
        if dataset == 'train_set':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(dataset == 'train_set'):
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = criterion(output, labels)

                # Backward 
                if dataset == 'train_set':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train_set', 'valid_set', 'test_set']}
        epoch_loss = running_loss / dataset_sizes[dataset]
        epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
        
        print("Epoch: {}/{}... ".format(e, epochs),
              "{} Loss: {:.4f}    Accurancy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))
        

# Validating the model

match = 0
total = 0
model.to(device)
with torch.no_grad():
    for data in dataloaders['train_set']:    
        inputts, labells = data
        inputts, labells = inputts.to(device), labells.to(device)
        output = model(inputts)
        _, predicted = torch.max(output.data, 1)
        total += labells.size(0)
        match += (predicted == labells).sum().item()
        
print(architecture, 'accuracy on the test inputs is: %d %%' % (100 * match / total))


# Saving Checkpoint
model.class_to_idx = image_datasets['train_set'].class_to_idx
model.to(device)
torch.save({'model': architecture,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
