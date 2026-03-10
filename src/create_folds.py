# CREATE_FOLDS.PY
import pandas as pd
from sklearn import model_selection 
from src import config 

if __name__== "__main__":

    df = pd.read_csv(config.TRAIN_FILE)
    
    df["kfold"] = -1
    
    # Randomize the rows of data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = df.label.values
    
    # Initiate Kfold
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Fill the new kfold column
    for n_fold, (train_,validation_) in enumerate(kf.split(X=df, y=y)):
        df.loc[validation_, 'kfold'] = n_fold 
    
    # Save csv with kfold column
    df.to_csv(config.TRAINING_FILE, index=False)