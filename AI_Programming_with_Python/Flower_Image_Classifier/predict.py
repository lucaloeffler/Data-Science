#The predicting script uses the checkpoint file and uses that for prediction
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
from collections import OrderedDict
import argparse
import json

parser = argparse.ArgumentParser (description = "Prediction script")
parser.add_argument ('--arch', help = 'Vgg16 can be used if provided with valid argument else densenet121 is used', type = str)
parser.add_argument ('--image_path', default='cat_to_name.json' ,help = ' Catagory to Name Json File', type = str)
parser.add_argument ('--load_checkpoint_dir', help = 'Provide saved model directory. (Optional)', type = str)
parser.add_argument ('--category_names', help = 'Provide JSON file name. (Optional)', type = str)
parser.add_argument ('--top_k', help = 'Top k number of classes', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU(mandatory)", type = str)
args = parser.parse_args ()


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def loading_checkpoint (file_path):
    checkpoint = torch.load (file_path) 
    if checkpoint ['arch'] == 'vgg16':
        model = models.vgg16 (pretrained = True)
        
    else: 

        model = models.densenet121(pretrained = True)
        
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']

    for param in model.parameters():
         param.requires_grad = False 

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    w,h=224,224
    img=Image.open(image)
    
    img=img.resize((256,256))
    
    left=(img.size[0]/2-w/2)
    
    top=(img.size[1]/2-h/2)
    
    right=left+224
    
    bottom=top+224
    
    img=img.crop((left,top,right,bottom))
    
    np_img=np.array(img)/255
    
    mean=np.array([0.485, 0.456, 0.406])
    
    std=np.array([0.229, 0.224, 0.225])
    
    np_img=(np_img-mean)/std
    
    tra_image=np_img.transpose((2,0,1))
    
    image = torch.from_numpy(tra_image)
    
    return image
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    
    image = image.numpy().transpose((1, 2, 0))
    
   
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    
    model.eval()
    
    with torch.no_grad():
        
        
        proc_img = process_image(image_path)

        
        proc_img =proc_img.type(torch.FloatTensor).to(device)
        
        proc_img.unsqueeze_(0)

        
        log_loss = model(proc_img)
        

        ps = torch.exp(log_loss)

        
        top_p, top_classes = ps.topk(topk, dim=1)
 
        top_p = top_p.tolist()[0]
    

        top_classes = top_classes.tolist()[0]
        
        index_to_class = {v:k for k, v in model.class_to_idx.items()}
        
        labels = []
        
        for tc in top_classes:
            labels.append(cat_to_name[index_to_class[tc]])
    
        return top_p, labels
    
    
device='cuda' if args.GPU=='GPU' else 'cpu'

        
my_model = loading_checkpoint (args.load_checkpoint_dir)

args.top_k=args.top_k if args.topk else 1

prob, classes = predict(args.image_path, my_model)
prob_dic={}
for flower in classes :
    for p in prob:
        prob_dic[flower]=p
print(prob_dic)