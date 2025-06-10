import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt

#LOADING TRAINING DATA:

#EMAIL PHISHING DATA:
def load_email_data():
    try:
        df = pd.read_csv("data/email_phishing.csv", encoding='latin-1')
        df['label'] = df['Email Type'].map({
            'Phishing Email': 'spam',
            'Safe Email': 'ham'
        })
        df['text'] = df['Email Text']
        df['type'] = 'Email'
        df = df[['label', 'text', 'type']]
        print(f"Email dataset loaded: {len(df)} messages")
        return df
    except Exception as e:
        print(f"Could not load Email data: {e}")
        return pd.DataFrame()
    
#SMS SPAM DATA:
def load_sms_data():
    try:
        df = pd.read_csv("data/sms_spam.csv", encoding='latin-1')
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
        df['type'] = 'SMS'
        print(f"SMS dataset loaded: {len(df)} messages")
        return df
    except Exception as e:
        print(f"Could not load SMS data: {e}")
        return pd.DataFrame()

print("IT IS RUNNING...")

#loading both datasets:
sms_df = load_sms_data()
email_df = load_email_data()


df = pd.concat([email_df, sms_df], ignore_index=True) #combining datasets

#cleaning data:
df = df.dropna()
df['text'] = df['text'].astype(str)

#encoding labels: ham = 0, spam = 1
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

#tokenizing:
tk = Tokenizer(num_words=10000, oov_token="<OOV>")
tk.fit_on_texts(df['text'])

#converting text to sequences:
sequences = tk.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=100)
y = df['label'].values


#splitting the data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #80% training, 20% testing

#building the model:
model = Sequential([Embedding(10000, 64, input_length=100), LSTM(32, dropout=0.3, recurrent_dropout=0.3), Dense(16, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')])

#using LSTM  - a type of NN used for sequential data, like text, speech, or time series.


#compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#training model:
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,  # stop after 4 epochs without improvement
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stop]
)
#evaluating model:
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

#saving the model and tokenizer to another file:
model.save('models/spam_detector.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tk, f) #using pickle to web app can tokenize incoming messages the same way

#plotting training data
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')
plt.show()

#testing with sample messages
test_messages = [
    #email phishing examples
    ("URGENT: Your account has been compromised. Click here immediately!", "Email"),
    ("Congratulations! You've won $10,000! Click here to claim now!", "Email"),
    ("Your PayPal account will be suspended. Verify here: http://fake-link.com", "Email"),
    
    #SMS spam examples  
    ("Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121", "SMS"),
    ("WINNER!! You have been selected to receive a £900 prize reward!", "SMS"),
    
    #real messages examples
    ("Hey, are we still meeting for lunch tomorrow?", "SMS"),
    ("Thanks for dinner last night, had a great time!", "SMS"),
    ("Meeting reminder: Project review tomorrow at 2 PM in Conference Room A", "Email"),
    ("Your package has been delivered. Thank you for choosing our service.", "Email")
]

for msg, msg_type in test_messages:
    seq = tk.texts_to_sequences([msg])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)[0][0]
    label = "SPAM" if pred > 0.5 else "HAM"
    print(f"[{msg_type}] '{msg[:50]}...' -> {label} (confidence: {pred:.3f})")