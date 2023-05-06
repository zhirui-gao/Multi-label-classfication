import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import re
import random
import torch
from torch import nn
import torchvision.models as models
from sklearn.metrics import precision_score,recall_score,f1_score
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
random.seed(10)
BASE_PATH = 'C:/Users/30214/Desktop/multi-classficaiton/secode/COMP5329S1A2Dataset'


def read_csv(path, n_columns=2):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if not re.match('^\d+\.jpg', line):
                continue
            ImageID = line.split(',')[0]
            if n_columns == 2:
                Labels = line.split(',')[1]
                data.append({'ImageID': ImageID, 'Labels': Labels})
            else:
                data.append({'ImageID': ImageID})

    return pd.DataFrame(data)

train_df = read_csv(f'{BASE_PATH}/train.csv')

test_df = read_csv(f'{BASE_PATH}/test.csv', 1)
# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


# Initialize the model
n_class = 19
model = Resnext50(n_class)
# Switch model to the training mode
model.train()
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 加载数据集
class MultiLabelDataset(Dataset):
    def __init__(self, data_df, data_split='train', transform=None):
        self.labels = []
        self.imgs_path =[]
        self.transform = transform
        for index, row in enumerate(data_df.itertuples()):
            # if index>100:
            #     break
            if data_split=='train':
                lables = row.Labels
            else:
                lables = '0'
            image_id = row.ImageID

            imgs_path = os.path.join(BASE_PATH,'data',image_id)

            self.imgs_path.append(imgs_path)


            lables = lables.strip()
            if len(lables) == 0:
                continue
            label = [int(x)-1 for x in lables.split()]
            label_tensor = torch.zeros(n_class)
            label_tensor[label]=1
            self.labels.append(label_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label


def main():
    batch_size = 8
    max_epoch_number = 35
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 0
    iteration = 1
    train_set = MultiLabelDataset(train_df, 'train', transform=transform)
    test_set = MultiLabelDataset(test_df, 'test', transform=transform)
    # 定义数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    while True:
        batch_losses = []
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets.type(torch.float))
            print('iteration  loss:',iteration,  loss,)
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            batch_losses.append(batch_loss_value)

            if iteration % 1000 == 0:
                data = []
                thr = 0.5
                model.eval()
                with torch.no_grad():
                    model_result = []
                    targets = []
                    for imgs, batch_targets in test_loader:
                        imgs = imgs.to(device)
                        model_batch_result = model(imgs)
                        model_result.extend(model_batch_result.cpu().numpy())
                        targets.extend(batch_targets.cpu().numpy())
                    index=0
                    for result in model_result:
                        pre_labels = np.where(result>thr)
                        data.append(
                            {'ImageID': f"{index + 30000}.jpg", 'Labels': '_'.join([str(label) for label in pre_labels])})
                        index+=1
                pd.DataFrame(data).to_csv(f'{BASE_PATH}/{iteration}_submission.csv', index=False)

                result = calculate_metrics(np.array(model_result), np.array(targets))
                print("epoch:{:2d} iter:{:3d} test: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f}".format(epoch, iteration,
                                                  result['micro/f1'],
                                                  result['macro/f1'],
                                                  result['samples/f1']))

                model.train()
            iteration += 1

        loss_value = np.mean(batch_losses)
        print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
        # if epoch % 10 == 0:
        #     checkpoint_save(model, save_path, epoch)
        epoch += 1
        if max_epoch_number < epoch:
            break

main()


data = []
# for index, row in enumerate(test_df.itertuples()):
#     n_labels = random.randrange(1,3)
#     labels = []
#     for ix in range (n_labels):
#         labels.append(rand_label())
#     labels.sort()
#     data.append({'ImageID': f"{index+30000}.jpg",'Labels':'_'.join([str(label) for label in labels])})

pd.DataFrame(data).to_csv(f'{BASE_PATH}/submission.csv',index=False)


print(train_df,test_df)

