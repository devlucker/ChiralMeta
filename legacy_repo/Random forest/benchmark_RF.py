import os
import time
import torch
import random
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from get_tasks import load_tasks
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_rf_model():
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []

    n_trials = 10
    n_support = 64
    n_query = 128

    print('\nBeginning training loop...')
    for i in range(0, n_trials):
        print(f'Starting trial {i}')
        for task in test_tasks:
            # sample a task
            data_support = train_dfs[task]
            data_query = test_dfs[task]
            scaler = StandardScaler()
            
            indices = list(np.arange(len(data_support)))
            np.random.shuffle(indices)
            support_indices = np.sort(indices[0:n_support])

            # support set
            X_support = data_support.iloc[:,0:1544]
            X_scaled_support = scaler.fit_transform(X_support)
            X_support = X_scaled_support[support_indices.astype(int)]
            X_support = np.array(X_support)
            y_support = (data_support['ee_class']).values.tolist()
            y_support = np.array(y_support)[support_indices.astype(int)]

            indices = list(np.arange(len(data_query)))
            np.random.shuffle(indices)
            query_indices = np.sort(indices[0:n_query])

            # query set
            X_query = data_query.iloc[:,0:1544]
            X_scaled_query = scaler.fit_transform(X_query)
            X_query = X_scaled_query[query_indices.astype(int)]
            X_query = np.array(X_query)
            y_q = (data_query['ee_class']).values.tolist()
            y_query = np.array(y_q)[query_indices.astype(int)]

            model = RandomForestClassifier(n_estimators=200)
            model.fit(X_support, y_support)
            preds_y = model.predict(X_query)
            
            task_labels.append(y_query)
            task_preds.append(preds_y)

        predictions = np.concatenate(task_preds, axis=0)
        labels=np.concatenate(task_labels, axis=0)  

        roc_auc = roc_auc_score(labels, predictions)
    
        prec=average_precision_score(labels, predictions)

        task_roc.append(roc_auc)
        task_prec.append(prec)

    task_roc = np.array(task_roc)
    task_prec = np.array(task_prec)
    
    print("ROC_score: {:.4f} +- {:.4f}\n".format(np.mean(task_roc), np.std(task_roc)))
    print("Avg_prec: {:.4f} +- {:.4f}\n".format(np.mean(task_prec), np.std(task_prec))) 

    return None


evaluate_rf_model()