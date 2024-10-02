from models.RepSNet_v1 import RepSNet_v1, model_convert
import numpy as np
import torch

torch.set_grad_enabled(False)


def inference():
    result = []
    images = np.load("images.npy")
    model = RepSNet_v1(deploy=True).to("cuda")
    model.load_state_dict(torch.load("/opt/algorithm/data/model.pkl"))
    model.eval()
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    for batch in images.split(4):
        
        batch = batch.cuda()

        nucleus_map = torch.zeros(batch.shape[0], 2, 256, 256).float().cuda()
        type_map = torch.zeros(batch.shape[0], 7, 256, 256).float().cuda()
        boundary_map = torch.zeros(batch.shape[0], 256, 256).float().cuda()
        
        ##########未做任何增强
        pred = model(batch)
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1)
        type_map += torch.softmax(pred["type_map"], dim=1)
        boundary_map += pred["boundary_map"]
        
        ##########水平翻转
        pred = model(batch.flip(-1))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).flip(-1)
        type_map += torch.softmax(pred["type_map"], dim=1).flip(-1)
        boundary_map += pred["boundary_map"].flip(-1)

        ##########上下翻转
        # pred = model(batch.flip(-2))
        # nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).flip(-2)
        # type_map += torch.softmax(pred["type_map"], dim=1).flip(-2)        
        # boundary_map += pred["boundary_map"].flip(-2)
        
       

        ##########旋转90度
        pred = model(batch.rot90(1, [-2, -1]))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).rot90(-1, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).rot90(-1, [-2, -1])
        boundary_map += pred["boundary_map"].rot90(-1, [-2, -1])

        ##########旋转90度 + 水平翻转
        pred = model(batch.rot90(1, [-2, -1]).flip(-1))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).flip(-1).rot90(-1, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).flip(-1).rot90(-1, [-2, -1])
        boundary_map += pred["boundary_map"].flip(-1).rot90(-1, [-2, -1])

        ##########旋转180度
        pred = model(batch.rot90(2, [-2, -1]))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).rot90(-2, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).rot90(-2, [-2, -1])
        boundary_map += pred["boundary_map"].rot90(-2, [-2, -1])
        
        ##########旋转180度 + 水平翻转
        pred = model(batch.rot90(2, [-2, -1]).flip(-1))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).flip(-1).rot90(-2, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).flip(-1).rot90(-2, [-2, -1])
        boundary_map += pred["boundary_map"].flip(-1).rot90(-2, [-2, -1])

        ##########旋转270度
        pred = model(batch.rot90(3, [-2, -1]))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).rot90(-3, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).rot90(-3, [-2, -1])
        boundary_map += pred["boundary_map"].rot90(-3, [-2, -1])
        
        ##########旋转270度 + 水平翻转
        pred = model(batch.rot90(3, [-2, -1]).flip(-1))
        nucleus_map += torch.softmax(pred["nucleus_map"], dim=1).flip(-1).rot90(-3, [-2, -1])
        type_map += torch.softmax(pred["type_map"], dim=1).flip(-1).rot90(-3, [-2, -1])
        boundary_map += pred["boundary_map"].flip(-1).rot90(-3, [-2, -1])
        
        pred = {"nucleus_map": nucleus_map, "type_map": type_map, "boundary_map": boundary_map}
        
        ann_pred = model.get_ann(pred)
        
        result.append(ann_pred)


    result = np.concatenate(result)
    # regression
    pred_regression = {
        "neutrophil": [],
        "epithelial-cell": [],
        "lymphocyte": [],
        "plasma-cell": [],
        "eosinophil": [],
        "connective-tissue-cell": [],
    }

    keys = ["", "neutrophil", "epithelial-cell", "lymphocyte", "plasma-cell", "eosinophil", "connective-tissue-cell"]

    for pred in result:
        for i in range(1, 7):
            pred = pred[16:-16, 16:-16]
            ids = pred[..., 0][pred[..., 1] == i]
            count = len(np.unique(ids))
            pred_regression[keys[i]].append(count)

    return result, pred_regression

