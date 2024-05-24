from sklearn.metrics import accuracy_score,roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.model import *
from data.load_data import *
from denoise import test_denoise
import torch
from tqdm import tqdm
from util.parameters import get_args
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from util.globals import *

def cls_model(args, model_num, denoise=False, save_model=False, model_ckpt=None):
    if denoise:
        train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs.pt'))
        test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs.pt'))
        train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt'))
        test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt'))
    else:
        train_embeddings = torch.load(os.path.join(save_dir, f'train_cls_embs.pt'))
        test_embeddings = torch.load(os.path.join(save_dir, f'test_cls_embs.pt'))
        train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt'))
        test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt'))
    train_embeddings  = train_embeddings.to(args.device)
    test_embeddings = test_embeddings.to(args.device)
    train_labels = train_labels.to(args.device)
    test_labels = test_labels.to(args.device)

    # initialize the classifier (user side)
    input_dim = train_embeddings.shape[-1]
    if model_num == 0:
        cls_model = EnhancedClsModel(input_dim, 2).to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    elif model_num == 1:
        cls_model = clsModel1(input_dim, 2).to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    elif model_num == 2:
        cls_model = clsModel2(input_dim, 2).to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        cls_model = clsModel(input_dim, 2).to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)
    if model_ckpt is not None:
        cls_model.load_state_dict(torch.load(model_ckpt))
    optimizer = optim.SGD(cls_model.parameters(), lr=0.001, momentum=0.9)

    noise_results = []

    # train the classifier (user side)
    epoch_accuracies = []
    epoch_losses = []
    epoch_aucs = []

    if model_ckpt is not None:
        y_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(train_embeddings), args.cls_batch_size)):
                y_logit = cls_model(test_embeddings[i:i+args.cls_batch_size])
                y_pred = torch.argmax(y_logit, -1)
                y_preds.append(y_pred)
        y_preds = torch.cat(y_preds)
        if 'cuda' in args.device:
            y_preds = y_preds.cpu()
            test_labels = test_labels.cpu()
        accuracy = accuracy_score(test_labels, y_preds)
        auc = roc_auc_score(test_labels, y_preds)
        print(f'Initial accuracy {accuracy}')

    for epoch in range(args.cls_epochs):
        for i in tqdm(range(0, len(train_embeddings), args.cls_batch_size)):
            Xbatch = train_embeddings[i:i+args.cls_batch_size]
            y_pred = cls_model(Xbatch)
            y_pred = y_pred.squeeze()
            ybatch = train_labels[i:i+args.cls_batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute the prediction and accuracy (user side)
        y_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(train_embeddings), args.cls_batch_size)):
                y_logit = cls_model(test_embeddings[i:i+args.cls_batch_size])
                y_pred = torch.argmax(y_logit, -1)
                y_preds.append(y_pred)
        y_preds = torch.cat(y_preds)
        if 'cuda' in args.device:
            y_preds = y_preds.cpu()
            test_labels = test_labels.cpu()
        accuracy = accuracy_score(test_labels, y_preds)
        auc = roc_auc_score(test_labels, y_preds)
        print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {accuracy}, auc {auc}')

        # Add accuracy and loss to lists
        epoch_accuracies.append(accuracy)
        epoch_aucs.append(auc)
        epoch_losses.append(loss.item())

    # Add results to overall results list
    noise_results.append({
        "accuracies": max(epoch_accuracies),
        "losses": epoch_losses,
        "aucs": max(epoch_aucs),
    })

    # Print results
    for result in noise_results:
        print(result)
    
    if save_model:
        cls_mode_subdir = os.path.join(args.denoise_model_dir, f"cls_model_{args.task}")
        if not os.path.exists(cls_mode_subdir):
            os.makedirs(cls_mode_subdir, exist_ok=True)
        out_path = os.path.join(cls_mode_subdir, f"cls_model_{model_num}")
        torch.save(cls_model.state_dict(), out_path)

def mlp_model(args, num):
    train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs.pt'))
    test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs.pt'))
    train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt'))
    test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt'))

    # initialize the classifier (user side)
    input_dim = train_embeddings.shape[-1]
    mlp_model = mlpModel(input_dim).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    # train the classifier (user side)
    epoch_accuracies = []
    epoch_losses = []
    for epoch in range(args.cls_epochs):
        optimizer.zero_grad()
        outputs = mlp_model(train_embeddings)
        loss = loss_fn(outputs, train_labels)
        loss.backward()
        optimizer.step()

    # Set the model to evaluation mode
    mlp_model.eval()

    # Initialize variables to keep track of predictions and correct predictions
    total_predictions = 0
    correct_predictions = 0

    # Disable gradient computation during testing
    with torch.no_grad():
        for i in range(len(test_embeddings)):
            # Get the test data and labels
            test_data = test_embeddings[i]
            true_label = test_labels[i]

            # Forward pass (predict the label)
            predicted_label = mlp_model(test_data.unsqueeze(0))  # Unsqueeze to add a batch dimension
            predicted_label = predicted_label.argmax().item()  # Get the index of the predicted class

            # Update the counts
            total_predictions += 1
            if predicted_label == true_label:
                correct_predictions += 1

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Set the model back to training mode
    mlp_model.train()

    # compute the prediction and accuracy (user side)
    y_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(train_embeddings), args.cls_batch_size)):
            y_logit = mlp_model(test_embeddings[i:i + args.cls_batch_size])
            y_pred = torch.argmax(y_logit, -1)
            y_preds.append(y_pred)
    y_preds = torch.cat(y_preds)
    if 'cuda' in args.device:
        y_preds = y_preds.cpu()
        test_labels = test_labels.cpu()
    accuracy = accuracy_score(test_labels, y_preds)
    print(f'Finished testing, accuracy {accuracy}')

    # Add accuracy and loss to lists
    epoch_accuracies.append(accuracy)
    epoch_losses.append(loss.item())

def mlp_model1(args):
    train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs.pt'))
    test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs.pt'))
    train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt'))
    test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt'))

    input_dim = train_embeddings.shape[-1]
    hidden_dims = [128, 64]
    mlp_model = mlpModel1(input_dim, hidden_dims, 2).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    #optimizer = optim.SGD(mlp_model.parameters(), lr=0.001, momentum=0.9)
    noise_results = []
    epoch_accuracies = []
    epoch_losses = []
    for epoch in range(args.cls_epochs):
        for i in tqdm(range(0, len(train_embeddings), args.cls_batch_size)):
            Xbatch = train_embeddings[i:i+args.cls_batch_size].to(args.device)
            ybatch = train_labels[i:i+args.cls_batch_size].to(args.device)

            y_pred = mlp_model(Xbatch)
            loss = loss_fn(y_pred, ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute prediction and accuracy
        y_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_embeddings), args.cls_batch_size)):
                Xbatch = test_embeddings[i:i+args.cls_batch_size].to(args.device)
                y_logit = mlp_model(Xbatch)
                y_pred = torch.argmax(y_logit, -1)
                y_preds.append(y_pred)
        
        y_preds = torch.cat(y_preds)
        if 'cuda' in args.device:
            y_preds = y_preds.cpu()
            test_labels = test_labels.cpu()

        accuracy = accuracy_score(test_labels, y_preds)
        print(f'Finished epoch {epoch}, latest loss {loss.item()}, accuracy {accuracy}')

        epoch_accuracies.append(accuracy)
        epoch_losses.append(loss.item())
    
    # Add results to overall results list
    noise_results.append({
        "accuracies": epoch_accuracies,
        "losses": epoch_losses
    })

    # Print results
    for result in noise_results:
        print(result)

def xgb_model(save_dir, learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=0.7, colsample_bytree=0.7):
    train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs.pt')).cpu().numpy()
    test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs.pt')).cpu().numpy()
    train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt')).cpu().numpy()
    test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt')).cpu().numpy()

    # Create DMatrix for train and test
    print("xgb:")
    dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    dtest = xgb.DMatrix(test_embeddings, label=test_labels)
    
    # Set parameters
    param = {
        'objective': 'multi:softmax',
        'num_class': 2,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }

    num_runs = 5
    accuracy_list = []

    for run in range(num_runs):
        # Optional: Change seed or other hyperparameters here
        param['seed'] = run  # Change the seed for each run

        # Train model
        bst = xgb.train(param, dtrain, 20)

        # Make prediction
        preds = bst.predict(dtest)

        # Evaluate model
        accuracy = accuracy_score(test_labels, preds)
        print(f'Test Accuracy for run {run + 1}: {accuracy}')
        
        accuracy_list.append(accuracy)

    # Calculate average accuracy over multiple runs
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f'Average Test Accuracy over {num_runs} runs using XGB: {average_accuracy}')

def xgb_model1(args, num): #grid_search 
    # Loading data
    train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs_{num}.pt')).cpu().numpy()
    test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs_{num}.pt')).cpu().numpy()
    train_labels = torch.load(os.path.join(save_dir, f'train_labels_{num}.pt')).cpu().numpy()
    test_labels = torch.load(os.path.join(save_dir, f'test_labels_{num}.pt')).cpu().numpy()

    # Grid Search for Hyperparameter Tuning
    print("Grid search:")
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.7, 0.8, 0.9, 1],
    }
    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=2)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3)
    grid_search.fit(train_embeddings, train_labels)
    best_params = grid_search.best_params_
    print(f"Best hyperparameters: {best_params}")

    # Set best parameters
    param = best_params
    param['objective'] = 'multi:softmax'
    param['num_class'] = 2

    # Create DMatrix for train and test
    print("XGB training:")
    dtrain = xgb.DMatrix(train_embeddings, label=train_labels)
    dtest = xgb.DMatrix(test_embeddings, label=test_labels)

    # Train and evaluate the model
    num_rounds = 100  # Number of boosting rounds
    early_stopping_rounds = 10  # Stop if no improvement for 10 rounds
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    bst = xgb.train(param, dtrain, num_rounds, evals=evals, early_stopping_rounds=early_stopping_rounds)

    # Make prediction using the best model
    best_iteration = bst.best_iteration  # Best iteration number
    preds = bst.predict(dtest, ntree_limit=best_iteration)

    # Evaluate model
    accuracy = accuracy_score(test_labels, preds)
    print(f'Test Accuracy: {accuracy}')

#random forest
def rf_model(args,num):
    train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs_{num}.pt')).cpu().numpy()
    test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs_{num}.pt')).cpu().numpy()
    train_labels = torch.load(os.path.join(save_dir, f'train_labels_{num}.pt')).cpu().numpy()
    test_labels = torch.load(os.path.join(save_dir, f'test_labels_{num}.pt')).cpu().numpy()
    
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, preds)
    print(f'Test Accuracy using Random Forest: {accuracy}')


#SVM
def svm_model(args):
    train_embeddings = torch.load(os.path.join(save_dir, 'denoise_train_cls_embs_1.pt')).cpu().numpy()
    test_embeddings = torch.load(os.path.join(save_dir, 'denoise_test_cls_embs_1.pt')).cpu().numpy()
    train_labels = torch.load(os.path.join(save_dir, 'train_labels_1.pt')).cpu().numpy()
    test_labels = torch.load(os.path.join(save_dir, 'test_labels_1.pt')).cpu().numpy()

    clf = SVC()
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, preds)
    print(f'Test Accuracy using SVM: {accuracy}')


#KNN
def knn_model(args):
    train_embeddings = torch.load(os.path.join(save_dir, 'denoise_train_cls_embs_1.pt')).cpu().numpy()
    test_embeddings = torch.load(os.path.join(save_dir, 'denoise_test_cls_embs_1.pt')).cpu().numpy()
    train_labels = torch.load(os.path.join(save_dir, 'train_labels_1.pt')).cpu().numpy()
    test_labels = torch.load(os.path.join(save_dir, 'test_labels_1.pt')).cpu().numpy()

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(train_embeddings, train_labels)
    preds = clf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, preds)
    print(f'Test Accuracy using KNN: {accuracy}')


if __name__ == "__main__":
    from denoise import test_denoise
    args = get_args()
    #load emb and labels
    # base_mod_name = args.base_model.split('/')[-1]
    # if args.noise_mechanism == "Gaussian":
    #     save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}")
    # elif args.noise_mechanism == "ChiDP":
    #     save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.eta}")
    save_dir = "/home/data/yrph/denoise/data/embedding_bert-large-uncased_imdb_denoiseModelv3_att_w_output_v3_True_True_True"
    #cls_model(args,0,2)
    # cls_mod_ckpt = "/home/data/yrph/denoise/model/cls_model_glue_sst2/cls_model_1"
    cls_model(args, model_num=0, denoise=True, save_model=False, model_ckpt=None)
    # xgb_model(save_dir)
    #keras_sequential(args,0)
    #rf_model(args)
    #svm_model(args)
    #knn_model(args)

    save_dir = "/home/data/yrph/denoise/data/embedding_glue_sst2_bert-base-uncased_glue_sst2_denoiseModelv3_att_w_output_v3_True_True_True"
    # denoise_train_embeddings = torch.load(os.path.join(save_dir, f'denoise_train_cls_embs.pt'))
    # denoise_test_embeddings = torch.load(os.path.join(save_dir, f'denoise_test_cls_embs.pt'))
    # train_labels = torch.load(os.path.join(save_dir, f'train_labels.pt'))
    # test_labels = torch.load(os.path.join(save_dir, f'test_labels.pt'))
    # save_dir = "/home/data/yrph/denoise/data/clean_cls_embedding_glue_sst2_bert-base-uncased"
    # clean_train_embeddings = torch.load(os.path.join(save_dir, f'train_cls_embs.pt'))
    # clean_test_embeddings = torch.load(os.path.join(save_dir, f'test_cls_embs.pt'))
    # this_noise_mse, this_denoise_mse, this_noise_sim, this_denoise_sim = test_denoise(clean_train_embeddings, denoise_train_embeddings, denoise_train_embeddings, args)
    # print(f"MSE after denoise: {this_denoise_mse}, Cosine similarity after denoise: {this_denoise_sim}")
    # this_noise_mse, this_denoise_mse, this_noise_sim, this_denoise_sim = test_denoise(clean_test_embeddings, denoise_test_embeddings, denoise_test_embeddings, args)
    # print(f"MSE after denoise: {this_denoise_mse}, Cosine similarity after denoise: {this_denoise_sim}")
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import Pipeline
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import PCA
    # pca = PCA()
    # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    # Xt = pipe.fit_transform(clean_train_embeddings.cpu())
    # plot = plt.scatter(Xt[:,0], Xt[:,1], c=train_labels.cpu())
    # plt.legend(handles=plot.legend_elements()[0], labels=[0,1])
    # plt.savefig('pca_sst2')