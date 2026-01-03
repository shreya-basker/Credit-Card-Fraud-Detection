import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
data=pd.read_csv('credit_card_fraud_10k.csv')
#print(data.head())
print("\n Dataset info")
print(data.info())
print("\n Statistical Summary")
print(data.describe())
sns.countplot(x='is_fraud',data=data)
plt.title("Fraud vs Normal Transactions")
plt.show()
X=data.drop('is_fraud',axis=1)
y=data['is_fraud']
print("Feature shape:",X.shape)
print("Label shape:",y.shape)
categorical_cols=X.select_dtypes(include=['object']).columns
X=pd.get_dummies(X,columns=categorical_cols,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training samples",X_train.shape)
print("Testing samples",X_test.shape)


# feature scaling - convert all variables into the same unit making it easy to understand
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test) # why are we using fit_transform in one and only transform in the other?

# training ML
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print("Model Training Complete")

# evaluate the model
y_pred=model.predict(X_test)
print("/n Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("/n Classification report")
print(classification_report(y_test,y_pred))

