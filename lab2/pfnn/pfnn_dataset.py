from torch.utils.data import Dataset
import h5py as h5

class PFNNDataset(Dataset):
    def __init__(self, file="pfnn_dataset.h5"):
        super().__init__()
        self.file = h5.File(file, 'r')
        self.x_dst = self.file.get('training/input')
        self.y_dst = self.file.get('training/output')
        self.p_dst = self.file.get("training/phase")
        self.len = self.x_dst.shape[0]

    def __getitem__(self, index):
        return [self.x_dst[index], self.y_dst[index], self.p_dst[index]]
    
    def __len__(self):
        return self.len
    
if __name__ == '__main__':
    dataset = h5.File("pfnn_dataset.h5", 'r')
    x_dst = dataset.get("training/input")
    y_dst = dataset.get("training/output")
    p_dst = dataset.get("training/phase")
    print("x_dst: {}, y_dst: {}, p_dst: {}".format(x_dst.shape, y_dst.shape, p_dst.shape))