from data_processing import preprocessing
from binary_classification import CustomDataset, ModelClass, bin_classifier

data = preprocessing('./Dataset/constraint_Hindi_Train - Sheet1_combined.csv')
arr = data.processed()


mod = bin_classifier(arr, "ai4bharat/indic-bert", 'non-hostile', 15, 1e-5)
mod.train_model()




