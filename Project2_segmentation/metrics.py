import matplotlib.pyplot as plt

class Metrics:
    def dice_overlap(TP, FP, FN):
        precision = (TP +1) / (TP + FP +1)
        recall = (TP + 1) / (TP + FN + 1)
        dice = 2/ ((1/precision) + (1/recall))
        # dice = (2*TP)/(2*TP+FP+FN)
        return dice
    
    def IOU(TP, FP, FN):
        return TP / (TP + FN + FP)
    
    def accuracy(TP, TN, FP, FN):
        return (TP + TN) / (TP + FP + TN +FN)
    
    def sensitivity(TP, FN):
        return TP / (TP + FN)
    
    def specificity(TN, FP):
        return TN / (FP + TN)
    
    def plot(tensor_vals, title):
        metric_vals = [value.cpu().data.numpy() for value in tensor_vals]
        plt.plot(range(len(metric_vals)), metric_vals)
        #plt.legend(('Accuracy'))
        plt.xlabel('Epoch number')
        plt.ylabel(title)
        plt.show()