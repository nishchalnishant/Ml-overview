---
module: Practice
topic: Coding Drills
subtopic: Debug Drills
status: unread
tags: [practice, coding, debugging, ml]
---
# Debug Drills

Twelve snippets. Each **runs without raising** and produces plausible-looking output.
Each contains exactly one bug that would matter in production. Find it, explain why it
is wrong, and fix it.

This is the closest drill to real ML work, and to the "here is a notebook, tell me
what's wrong with it" interview round.

**Solutions:** [03-debug-solutions.md](03-debug-solutions.md).

**Rules.** 10 minutes each. Write down the bug *before* checking. If you cannot find
it in 10 minutes, note what you ruled out — that list is the useful artifact.

---

### D1

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)
model.fit(X_tr, y_tr)
print("test acc:", model.score(X_te, y_te))
```

### D2

```python
for epoch in range(epochs):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} loss {loss.item():.4f}")
```

### D3

```python
def evaluate(model, loader):
    correct = 0
    for xb, yb in loader:
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
    return correct / len(loader)
```

### D4

```python
df["rolling_mean_7d"] = df.groupby("user_id")["amount"].transform(
    lambda s: s.rolling(7).mean()
)
df["target"] = df["is_fraud"]
model.fit(df[["rolling_mean_7d"]], df["target"])
```

### D5

```python
class Net(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 128)
        self.fc2 = nn.Linear(128, d_out)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.softmax(self.fc2(x), dim=1)

criterion = nn.CrossEntropyLoss()
```

### D6

```python
from sklearn.model_selection import cross_val_score, KFold

scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True),
                         scoring="accuracy")
print(f"CV accuracy: {scores.mean():.3f}")
# X is time-ordered transaction data; y is next-day churn
```

### D7

```python
def train_step(model, xb, yb, opt):
    opt.zero_grad()
    out = model(xb)
    loss = loss_fn(out, yb)
    loss.backward()
    opt.step()
    losses.append(loss)
    return loss

# after 10k steps, memory usage grows without bound
```

### D8

```python
embeddings = model.encode(documents)
query_emb = model.encode([query])[0]
sims = embeddings @ query_emb
top = np.argsort(sims)[:10]
return [documents[i] for i in top]
```

### D9

```python
from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE().fit_resample(X, y)
scores = cross_val_score(model, X_res, y_res, cv=5, scoring="f1")
print(f"F1: {scores.mean():.3f}")   # 0.94 -- great!
# production F1 is 0.31
```

### D10

```python
def predict_batch(texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**tokens).logits
    return logits.argmax(-1).tolist()
# single-item predictions differ from the same item inside a large batch
```

### D11

```python
best_score, best_params = 0, None
for params in param_grid:
    model = build(params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score, best_params = score, params
print(f"best params {best_params}, expected accuracy {best_score:.3f}")
```

### D12

```python
class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.n = 0
    def update(self, batch_loss, batch_size):
        self.total += batch_loss
        self.n += batch_size
    @property
    def value(self):
        return self.total / self.n
# reported epoch loss is far lower than any individual batch loss
```
