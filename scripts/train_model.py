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
    def __init__(self, texts, labels, rewards, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.rewards = rewards
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

        reward = self.rewards[item]

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'reward': torch.tensor(reward, dtype=torch.float)
        }

def train_model():
    # 1. 載入對齊後的數據
    DATA_PATH = "datasets/processed/aligned_training_data.csv"
    if not os.path.exists(DATA_PATH):
        print("Error: 找不到對齊後的數據檔案。請先執行 scripts/align_market_data.py")
        return

    df = pd.read_csv(DATA_PATH)
    # 我們保留原本的分類目標，但加入 RL Reward 作為訓練權重
    texts = df['text'].values
    labels = df['is_contrarian_win'].values
    
    # 讀取 RL Reward (如果是舊資料沒有這欄，預設塞 0.0)
    if 'reward_pct' in df.columns:
        rewards = df['reward_pct'].fillna(0.0).values
    else:
        rewards = [0.0] * len(df)

    # 2. 分割訓練集與測試集
    train_texts, val_texts, train_labels, val_labels, train_rewards, val_rewards = train_test_split(
        texts, labels, rewards, test_size=0.2, random_state=42
    )

    # 3. 初始化 Tokenizer 與模型 (使用繁體中文預訓練模型)
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
    model = model.to(device)

    # 4. 準備 DataLoader
    train_data = BaniniDataset(train_texts, train_labels, train_rewards, tokenizer)
    val_data = BaniniDataset(val_texts, val_labels, val_rewards, tokenizer)
    
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
            rewards = batch['reward'].to(device)

            # 取得未經 softmax 的 Logits
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 自定義 RL 獎勵加權 Loss 計算
            # 若 Reward 越高 (代表 Agent 決策帶來極大虧損或極大獲利)，我們放大該樣本的 Loss 權重
            # 讓神經網路「深刻記住」這次教訓
            weights = 1.0 + torch.abs(rewards) * 10.0 # 報酬率每 1% 放大 0.1 倍學習力道
            base_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
            
            loss = (base_loss * weights).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(rl_loss=loss.item())

    # 7. 儲存模型
    MODEL_SAVE_PATH = "models/banini_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"訓練完成！模型已儲存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
