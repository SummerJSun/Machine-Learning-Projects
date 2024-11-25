def main():         
    import pandas as pd
    import numpy as np
    
    data = pd.read_csv('data.csv')
    data['person ID'] = data['person ID'].astype(int)

    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=66)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    specificities = []
    sensitivities = []
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        fold_train = data.iloc[train_idx]
        fold_val = data.iloc[val_idx]

        fold_train['person ID'].astype(int).to_csv('para2_file.txt', index=False, header=False)
        fold_val['person ID'].astype(int).to_csv('para3_file.txt', index=False, header=False)
        
        import decisiontree
        tree = decisiontree.DecisionTree()  
        tree.decision_tree('data.csv', 'para2_file.txt', 'para3_file.txt', 'para4_file.txt')
        
        predictions_df = pd.read_csv('para4_file.txt', 
                                   header=None, 
                                   names=['person ID', 'prediction'],
                                   sep='\t',
                                   dtype={'person ID': int})
        
        val_results = fold_val[['person ID', 'Has heart disease? (Prediction Target)']].merge(
            predictions_df, 
            on='person ID', 
            how='left'
        )
        
        true_labels = val_results['Has heart disease? (Prediction Target)'].values
        predictions = val_results['prediction'].values
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, pos_label='Yes')
        recall = recall_score(true_labels, predictions, pos_label='Yes')
        f1 = f1_score(true_labels, predictions, pos_label='Yes')
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=['No', 'Yes']).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        specificities.append(specificity)
        sensitivities.append(sensitivity)
        
        fold_results.append({
            'Fold': f'Fold {fold + 1}',
            'Accuracy': f'{accuracy:.3f}',
            'Precision': f'{precision:.3f}',
            'Recall/Sensitivity': f'{recall:.3f}',
            'F1-measure': f'{f1:.3f}',
            'Specificity': f'{specificity:.3f}'
        })

    fold_df = pd.DataFrame(fold_results)
    
    overall_results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall/Sensitivity', 'F1-measure', 'Specificity'],
        'Mean': [
            f'{sum(accuracies)/k_folds:.3f}',
            f'{sum(precisions)/k_folds:.3f}',
            f'{sum(recalls)/k_folds:.3f}',
            f'{sum(f1_scores)/k_folds:.3f}',
            f'{sum(specificities)/k_folds:.3f}'
        ],
        'Std Dev': [
            f'±{np.std(accuracies):.3f}',
            f'±{np.std(precisions):.3f}',
            f'±{np.std(recalls):.3f}',
            f'±{np.std(f1_scores):.3f}',
            f'±{np.std(specificities):.3f}'
        ]
    })

    print("\nResults for Each Fold:")
    print(fold_df.to_string(index=False))
    print("\nOverall Results:")
    print(overall_results.to_string(index=False))

if __name__ == "__main__":
    main()





