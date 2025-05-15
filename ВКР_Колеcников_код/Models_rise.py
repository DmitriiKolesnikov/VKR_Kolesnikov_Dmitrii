import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data_rise.csv', encoding='utf-8')

df = df.dropna()
num_cols = ['Норматив достаточности капитала (Н1)', 'Доля просроченных кредитов (NPL%)',
            'Рентабельность активов (ROA%)', 'Коэффициент ликвидности (Н3%)',
            'Размер банка (лог активов)', 'Доля банка в активах системы',
            'Темп роста кредитного портфеля (YoY)', 'Loan-to-Deposit Ratio (LDR, %)',
            'Доля межбанковских заимствований в пассивах (%)', 'Прирост прибыли банка (YoY)']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

cols_to_drop = ['Год', 'Квартал', 'Название банка', 'Этап экономического цикла',]
df.drop(columns=cols_to_drop, inplace=True)

df["Ключевая ставка в квадрате"] = df['Ключевая ставка'] ** 2
df['Норматив достаточности капитала (Н1) (log)'] = np.log(df['Норматив достаточности капитала (Н1)'])

features_log = ['Инфляция',
    'Темп прироста ВВП',
    'Ключевая ставка в квадрате',
    'Норматив достаточности капитала (Н1) (log)',
    'Доля просроченных кредитов (NPL%)',
    'Рентабельность активов (ROA%)',
    'Коэффициент ликвидности (Н3%)',
    'Размер банка (лог активов)',
    'Доля банка в активах системы',]

features_rf = features_log + ['Темп роста кредитного портфеля (YoY)',
                              'Loan-to-Deposit Ratio (LDR, %)',
                              'Доля межбанковских заимствований в пассивах (%)',
                              'Прирост прибыли банка (YoY)', 'Тип собственности банка']
X_rf = df[features_rf].copy()
X_rf = pd.get_dummies(X_rf, columns=['Тип собственности банка'], drop_first=False)
if 'Тип собственности банка_Частный' in X_rf.columns:
    X_rf = X_rf.drop(columns=['Тип собственности банка_Частный'])
y = df['Дефолт']

X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.2, random_state=42, stratify=y)

# Логистическая регрессия
X_logit = X_train[features_log]
X_logit = sm.add_constant(X_logit)
model_full = sm.Logit(y_train, X_logit).fit(disp=0)
print(model_full.summary())

features_center = ['Инфляция',
                   'Доля просроченных кредитов (NPL%)',
                   'Коэффициент ликвидности (Н3%)',
                   'Размер банка (лог активов)']

X_logit2 = X_train[features_center].copy()
X_logit2_centered = X_logit2 - X_logit2.mean()
X_logit2_centered = sm.add_constant(X_logit2_centered)
model_logit = sm.Logit(y_train, X_logit2_centered).fit(disp=0)
print(model_logit.summary())

X_test_logit = sm.add_constant(X_test[['Инфляция', 'Доля просроченных кредитов (NPL%)',
                                       'Коэффициент ликвидности (Н3%)', 'Размер банка (лог активов)']])
y_pred_prob_logit = model_logit.predict(X_test_logit)
y_pred_logit = (y_pred_prob_logit >= 0.5).astype(int)

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_prob_rf = rf.predict_proba(X_test)[:,1]
y_pred_rf = rf.predict(X_test)
importances = rf.feature_importances_
feature_names = X_train.columns

imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

total = imp_series.sum()
print("\nВажность признаков (RandomForest):")
for feat, imp in imp_series.items():
    print(f"{feat} – {imp/total*100:.1f}%")

# Метрики
acc_log = accuracy_score(y_test, y_pred_logit)
prec_log = precision_score(y_test, y_pred_logit, zero_division=1)
rec_log = recall_score(y_test, y_pred_logit, zero_division=1)
f1_log = f1_score(y_test, y_pred_logit)
auc_log = roc_auc_score(y_test, y_pred_prob_logit)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_prob_rf)

print(f"LogReg: accuracy={acc_log:.3f}, precision={prec_log:.3f}, recall={rec_log:.3f}, F1={f1_log:.3f}, AUC={auc_log:.3f}")
print(f"RandomForest: accuracy={acc_rf:.3f}, precision={prec_rf:.3f}, recall={rec_rf:.3f}, F1={f1_rf:.3f}, AUC={auc_rf:.3f}")

# ROC-кривые
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_logit)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
plt.figure(figsize=(6,6))
plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC={auc_log:.2f})', color='orange')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})', color='red')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend(); plt.grid(True)
plt.savefig('roc_curve.png')

# Матрицы ошибок
cm_log = confusion_matrix(y_test, y_pred_logit)
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig, axes = plt.subplots(1,2, figsize=(8,3))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set(title='Logistic Regression\nConfusion Matrix', xlabel='Предсказано', ylabel='Истинно')
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1])
axes[1].set(title='Random Forest\nConfusion Matrix', xlabel='Предсказано', ylabel='Истинно')
plt.savefig('confusion_matrices.png')

# Классификация по уровням риска
risk = pd.cut(y_pred_prob_logit, bins=[0, 0.1, 0.5, 1.0], labels=['Low','Medium','High'], right=False)
print("Распределение по уровням риска (LogReg):")
print(risk.value_counts().sort_index())

risk_rf = pd.cut(
    y_pred_prob_rf,
    bins=[0, 0.1, 0.5, 1.0],
    labels=['Low', 'Medium', 'High'],
    right=False
)
print("Распределение по уровням риска (RandomForest):")
print(risk_rf.value_counts().sort_index())
