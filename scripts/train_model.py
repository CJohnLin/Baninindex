import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# 設定裝置：優先使用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

class BaniniDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # 1. 載入對齊後的數據
    DATA_PATH = "datasets/processed/aligned_training_data.csv"
    if not os.path.exists(DATA_PATH):
        print("Error: 找不到對齊後的數據檔案。請先執行 scripts/align_market_data.py")
        return

    df = pd.read_csv(DATA_PATH)
    # 我們這裡將「反指標成功」作為分類目標 (0 或 1)
    # 也可以改為回歸預測報酬率，但分類通常在初期更穩定
    texts = df['text'].values
    labels = df['is_contrarian_win'].values

    # 2. 分割訓練集與測試集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 3. 初始化 Tokenizer 與模型 (使用繁體中文預訓練模型)
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
    model = model.to(device)

    # 4. 準備 DataLoader
    train_data = BaniniDataset(train_texts, train_labels, tokenizer)
    val_data = BaniniDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    # 5. 設定優化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    # 6. 訓練迴圈
    print("開始訓練...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())

    # 7. 儲存模型
    MODEL_SAVE_PATH = "models/banini_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"訓練完成！模型已儲存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
