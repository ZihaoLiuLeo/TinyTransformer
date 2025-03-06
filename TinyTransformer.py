import torch
import torch.nn as nn
import torch.optim as optim
import math

train_data = [
    ("我 爱 机器 学习", "I love machine learning"),
    ("今天 天气 很好", "Today the weather is nice"),
    ("你 吃饭 了 吗", "Have you eaten yet")
]

src_vocab = {'<pad>':0, '<unk>':1, '我':2, '爱':3, '机器':4, '学习':5, 
            '今天':6, '天气':7, '很好':8, '你':9, '吃饭':10, '了':11, '吗':12}
tgt_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3, 
            'I':4, 'love':5, 'machine':6, 'learning':7,
            'Today':8, 'the':9, 'weather':10, 'is':11, 'nice':12,
            'Have':13, 'you':14, 'eaten':15, 'yet':16}

def get_word_by_idx(vocab_dict, idx):
    return next(filter(lambda i:vocab_dict[i]==idx, vocab_dict.keys()), None)

def text_to_tensor(text, vocab, is_target=False):
    tokens = text.split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if is_target:
        indices = [vocab['<sos>']] + indices + [vocab['<eos>']]
    return torch.tensor(indices, dtype=torch.long)

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class TinyTransformer(nn.Module):
    def __init__(self, src_vocab_size=13, tgt_vocab_size=17, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src.permute(1,0,2),
                                  tgt.permute(1,0,2),
                                  tgt_mask=self.transformer.generate_square_subsequent_mask(tgt.size(1)))
        return self.fc(output.permute(1,0,2))

def generate_data(batch_size=32, seq_len=10):
    src = torch.randint(1,99,(batch_size, seq_len))
    tgt = torch.cat([src[:,:1], src[:,:-1]], dim=1)
    return src, tgt

def generate_batch(batch_size=3):
    src_batch, tgt_batch = [], []
    for src_text, tgt_text in train_data:
        src = text_to_tensor(src_text, src_vocab)
        tgt = text_to_tensor(tgt_text, tgt_vocab, is_target=True)
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    return src_padded.T, tgt_padded.T

if __name__ == '__main__':
    model = TinyTransformer(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        src, tgt = generate_batch()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, len(tgt_vocab)), tgt[:,1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    test_src = text_to_tensor("今天 天气 很好", src_vocab).unsqueeze(0)
    test_tgt_init = torch.tensor([[tgt_vocab['<sos>']]], dtype=torch.long)

    for _ in range(10):
        output = model(test_src, test_tgt_init)
        next_word = output.argmax(-1)[:,-1]
        test_tgt_init = torch.cat([test_tgt_init, next_word.unsqueeze(1)], dim=1)
        if next_word == tgt_vocab['<eos>']:
            break
    
    print("翻译结果:", " ".join(get_word_by_idx(tgt_vocab, idx) for idx in test_tgt_init[0]))
        

