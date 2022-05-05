from Util import *

# User Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--checkpoint_path')
parser.add_argument('--top_k')
parser.add_argument('--cate_names')
parser.add_argument('--device')
args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint_path
top_k = args.top_k
cate_names = args.cate_names
device = args.device

# If user inputs no arguments
device = "cpu"if device is None else device
top_k = 5 if top_k is None else top_k
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f) if cate_names is None or os.path.splitext(cate_names)[1] == '.json' else print('Please enter a json file')


def load_checkpoint(checkpoint_path):
#     checkpoint_path = torch.load("checkpoint.pth")

    architecture = "vgg13"
    input_, hidden_layers, output_ = 25088, 4096, 102 if architecture == "vgg13" or achitecture == "vgg19_bn" else print("what model do you want to use")
    model = models.vgg13(pretrained=True) if architecture == "vgg13" else models.vgg19_bn(pretrained=True) if architecture == "vgg19_bn" else print("Model not available")

#     model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint_path['class_to_idx']


    model.classifier = nn.Sequential(nn.Linear(input_, hidden_layers),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_layers, output_),
                                     nn.LogSoftmax(dim=1))


    model.load_state_dict(checkpoint_path['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    from PIL import Image
    import numpy as np
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    edit = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    convert_to_tensor = edit(img)
    processed_img = np.array(convert_to_tensor)
    processed_img = processed_img.transpose((0,2,1))
    
    return processed_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.class_to_idx = image_datasets['train_set'].class_to_idx
    model.to(device)
    torch_image = process_image(image_path)
    torch_image = torch.from_numpy(torch_image).type(torch.FloatTensor)
    torch_image = torch_image.unsqueeze(0).float()
    
    with torch.no_grad():
        output = model.forward(torch_image.to(device))
        
    prob = F.softmax(output.data, dim=1)
    prob = prob.topk(topk)
    score = np.array(prob[0][0])
    index = 1
    Flower_list = [cat_to_name[str(index + 1)] for index in np.array(prob[1][0])]
    
    return score, Flower_list

# TODO: Display an image along with the top 5 classes
def display(image_path, model):
    
    # Setting plot area
    plt.figure(figsize = (3,6))
    ax = plt.subplot(2,1,1)
    
    # Display test flower
    image = process_image(image_path)
    title  = image_path.split('/')
    print(cat_to_name[title[2]])
    imshow(image, ax, title = cat_to_name[title[2]]);
    
    # Making prediction
    score, flowers_list = predict(image_path, model) 
    fig,ax = plt.subplots(figsize=(4,3))
    sticks = np.arange(len(flowers_list))
    ax.barh(sticks, score, height=0.3, linewidth=2.0, align = 'center')
    ax.set_yticks(ticks = sticks)
    ax.set_yticklabels(flowers_list)
    
    

model = load_checkpoint(checkpoint_path)
image_path = 'flowers/test/1/image_06743.jpg'
image = process_image(image_path)
print(image.shape) 

# imshow(process_image("flowers/test/1/image_06752.jpg"))


image_path = "flowers/test/1/image_06752.jpg"
display(image_path, model)


image_path = 'flowers/test/1/image_06752.jpg'
display(image_path, model)




