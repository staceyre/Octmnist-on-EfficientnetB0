import torch 
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torchvision import transforms
from medmnist import OCTMNIST,INFO
import matplotlib.pyplot as plt
from PIL import Image
import d2l_self as d2l
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),  # ✅ 转为 3 通道灰度图
    transforms.RandomHorizontalFlip(),  # 水平翻转,进行数据增强
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data_flag='octmnist'
download=True
batch_size=32
num_classes=4
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
train_data=OCTMNIST(split='train',transform=train_transform,download=download)
val_data=OCTMNIST(split='val',transform=train_transform,download=download)
train_subset=Subset(train_data,range(2000))
val_subset=Subset(val_data,range(500))
train_loader=DataLoader(train_subset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
val_loader=DataLoader(val_subset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)

weights=EfficientNet_B0_Weights.DEFAULT
model=efficientnet_b0(weights=weights)
model.classifier[0]=nn.Dropout(0.4)
model.classifier[1]=nn.Linear(model.classifier[1].in_features,num_classes)
model.to(device)

def train_one_epoch(model,loader,criterion,optimizer):
    model.train()
    total_Loss,correct,total=0.0,0,0
    for images,labels in loader:
        images,labels=images.to(device),labels.squeeze().long().to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_Loss+=loss.item()
        _,predicted=outputs.max(1)
        correct+=predicted.eq(labels).sum().item()
        total+=labels.size(0)
    return total_Loss/len(loader),100.*correct/total
def evaluate(model,loader):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for images,labels in loader:
            images,labels=images.to(device),labels.squeeze().long().to(device)
            outputs=model(images)
            _,predicted=outputs.max(1)
            correct+=predicted.eq(labels).sum().item()
            total+=labels.size(0)
    return 100.*correct/total



criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)
train_loss_list=[]
train_acc_list=[]
val_acc_list=[]
best_val_acc=0.0
num_epochs=30
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
for epoch in range(0,num_epochs+1):
    print(f'epoch {epoch+1}:')
    train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer)
    val_acc=evaluate(model,val_loader)
    print(f'Train acc:{train_acc:.2f}% \t Val acc:{val_acc:.2f}%')
    animator.add(epoch + 1, (None, None, val_acc))
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        torch.save(model.state_dict(),'efficientnet_best.pth')
        print('最佳模型已保存')
    # ✅ 每轮训练后收集数据
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)

    epochs = list(range(1, len(train_acc_list) + 1))
    plt.plot(epochs, train_acc_list, label='Train Acc')
    plt.plot(epochs, val_acc_list, label='Val Acc')
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training/Validation Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curve.png')  # ✅ 保存图像


     