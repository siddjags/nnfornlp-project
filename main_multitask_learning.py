from data_processing import preprocessing
from multitask_learning import CustomDataset, ModelClass, multitask_classifier

data = preprocessing('./Dataset/constraint_Hindi_Train - Sheet1_combined.csv')
arr = data.processed()

mod1 = multitask_classifier(arr, "ai4bharat/indic-bert", 'non-hostile', 15, 1e-5)
mod1.train_model()

mod2 = multitask_classifier(arr, "ai4bharat/indic-bert", 'defamation', 15, 1e-5)
mod2.train_model()

mod3 = multitask_classifier(arr, "ai4bharat/indic-bert", 'fake', 15, 1e-5)
mod3.train_model()

mod4 = multitask_classifier(arr, "ai4bharat/indic-bert", 'hate', 15, 1e-5)
mod4.train_model()

mod5 = multitask_classifier(arr, "ai4bharat/indic-bert", 'offensive', 15, 1e-5)
mod5.train_model()