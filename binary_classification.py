import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


def print_confusion_matrix(X_train, X_test, y_train, y_test, model):
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     # display_labels=[0,1],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize,
                                     labels=[0, 1])
        disp.ax_.set_title(title)

    plt.show()
    print('Classification report train data')
    print(classification_report(y_train, model.predict(X_train)))
    print('Classification report test data')
    print(classification_report(y_test, model.predict(X_test)))


def print_feature_importance(model, X_train):
    feature_importances = pd.DataFrame(model.feature_importances_ * 100,
                                       # index = preprocessor.get_feature_names(),
                                       columns=['importance']).sort_values('importance',
                                                                           ascending=False)

    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(50).plot(kind='barh')


def print_confusion_matrix_with_dots(X_test, y_test, model):
    import matplotlib.pyplot as plt
    import seaborn as sns
    probability_columns = model.predict_proba(X_test)
    prediction_column = model.predict(X_test)

    prediction_column_df = pd.DataFrame(data=prediction_column, columns=["prediction"])
    probability_columns_df = pd.DataFrame(data=probability_columns, columns=["prob_renew", "prob_churn"])

    prediction = X_test.join(y_test, how='inner')

    prediction = pd.concat([prediction, prediction_column_df.set_index(X_test.index)], axis=1)
    prediction = pd.concat([prediction, probability_columns_df.set_index(X_test.index)], axis=1)

    random = pd.DataFrame(np.random.rand(prediction.shape[0], 1), columns=['random_numbers'])
    random['random_numbers'] = random['random_numbers'] / 2

    df_vi = pd.concat([prediction, random.set_index(y_test.index)], axis=1).reset_index()
    df_vi.prediction = df_vi.prediction.astype(float)
    df_vi.random_numbers = df_vi.random_numbers.astype(float)

    df_vi['prediction_visualization'] = df_vi['prediction'] + df_vi['random_numbers']
    df_vi.loc[df_vi['prediction_visualization'] > 1, 'prediction_visualization'] = df_vi[
                                                                                       'prediction_visualization'] - 0.5

    plt.axvline(0.50, 0, 1, linewidth=4, color='g')
    plt.axvline(0.6, 0, 1, linewidth=4, color='r')

    sns.scatterplot(data=df_vi, x="prob_churn", y="prediction_visualization", hue="label")


def binary_performances(X, y, model, thresh=0.5, labels=['Positives', 'Negatives']):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve, f1_score

    y_true = y
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    shape = y_prob.shape
    if len(shape) > 1:
        if shape[1] > 2:
            raise ValueError('A binary class problem is required')
        else:
            y_prob = y_prob[:, 1]

    plt.figure(figsize=[15, 10])

    # CONFUSION MATRIX
    # -------------------------------------------

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1)

    plt.subplots_adjust(wspace=.3, hspace=.3)

    cm = confusion_matrix(y_true, (y_prob > thresh).astype(int))

    ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, annot_kws={"size": 14}, fmt='g')
    cmlabels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

    for i, t in enumerate(ax.texts):
        t.set_text(t.get_text() + "\n" + cmlabels[i])

    plt.title('Confusion Matrix', size=15)
    plt.xlabel('Predicted Values', size=13)
    plt.ylabel('True Values', size=13)

    # Distributions of Predicted Probabilities of both classes
    # ----------------------------------------------------------------

    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=1)

    m_precision, m_recall, _ = precision_recall_curve(y_true, y_prob)
    m_f1, m_auc = f1_score(y_true, y_pred), auc(m_recall, m_precision)

    # plot the precision-recall curves
    no_skill = len(y_true[y_true == 1]) / len(y_true)

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill Model')
    plt.plot(m_recall, m_precision, label='Model', color='orange', lw=1)

    plt.xlabel('Recall', size=13)
    plt.ylabel('Precision', size=13)
    plt.title('Precision-Recall Curve', size=15)
    plt.legend(loc="lower left")

    # ROC curve with annotated decision point
    # ----------------------------------------------

    ax3 = plt.subplot2grid((2, 3), (0, 2), colspan=1)

    fp_rates, tp_rates, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fp_rates, tp_rates)

    plt.plot(fp_rates, tp_rates, color='orange', lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

    tn, fp, fn, tp = [i for i in cm.ravel()]

    plt.plot(fp / (fp + tn), tp / (tp + fn), 'bo', markersize=8, label='Decision Point')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', size=13)
    plt.ylabel('True Positive Rate', size=13)
    plt.title('ROC Curve', size=15)
    plt.legend(loc="lower right")

    # 2D probability map
    ax5 = plt.subplot2grid((2, 3), (1, 0), colspan=2)

    prediction_column_df = pd.DataFrame(data=model.predict(X), columns=["prediction"])
    probability_columns_df = pd.DataFrame(data=model.predict_proba(X), columns=["prob_renew", "prob_churn"])
    #TODO refactor duplicate code
    prediction = X.join(y, how='inner')

    prediction = pd.concat([prediction, prediction_column_df.set_index(X.index)], axis=1)
    prediction = pd.concat([prediction, probability_columns_df.set_index(X.index)], axis=1)

    random = pd.DataFrame(np.random.rand(prediction.shape[0], 1), columns=['random_numbers'])
    random['random_numbers'] = random['random_numbers'] / 2

    df_vi = pd.concat([prediction, random.set_index(y.index)], axis=1).reset_index()
    df_vi.prediction = df_vi.prediction.astype(float)
    df_vi.random_numbers = df_vi.random_numbers.astype(float)

    df_vi['prediction_visualization'] = df_vi['prediction'] + df_vi['random_numbers']
    df_vi.loc[df_vi['prediction_visualization'] > 1, 'prediction_visualization'] = df_vi[
                                                                                       'prediction_visualization'] - 0.5

    plt.axvline(0.50, 0, 1, linewidth=4, color='g')
    plt.axvline(0.80, 0, 1, linewidth=4, color='r')
    plt.title('Distributions of Predictions 2D', size=15)
    sns.scatterplot(data=df_vi, x="prob_churn", y="prediction_visualization", hue="label")

    ax4 = plt.subplot2grid((2, 3), (1, 2))

    plt.hist(y_prob[y_true == 1], density=True, bins=25,
             alpha=.5, color='green', label=labels[0])
    plt.hist(y_prob[y_true == 0], density=True, bins=25,
             alpha=.5, color='red', label=labels[1])
    plt.axvline(thresh, color='blue', linestyle='--', label='Boundary')
    plt.xlim([0, 1])
    plt.title('Distributions of Predictions', size=15)
    plt.xlabel('Positive Probability (predicted)', size=13)
    plt.ylabel('Samples (normalized scale)', size=13)
    plt.legend(loc="upper right")

    plt.show()

    tn, fp, fn, tp = [i for i in cm.ravel()]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)
    results = {
        "Precision": precision, "Recall": recall,
        "F1 Score": F1, "AUC": roc_auc
    }

    prints = [f"{kpi}: {round(score, 3)}" for kpi, score in results.items()]
    prints = ' | '.join(prints)
    print(prints)

    return results
