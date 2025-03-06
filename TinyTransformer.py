import torch
import torch.nn as nn
import torch.optim as optim
import math

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
    def __init__(self, vocab_size=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src.permute(1,0,2),
                                  tgt.permute(1,0,2),
                                  tgt_mask=self.transformer.generate_square_subsequent_mask(tgt.size(1)))
        return self.fc(output.permute(1,0,2))

def generate_data(batch_size=32, seq_len=10):
    src = torch.randint(1,99,(batch_size, seq_len))
    tgt = torch.cat([src[:,:1], src[:,:-1]], dim=1)
    return src, tgt

if __name__ == '__main__':
    model = TinyTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        src, tgt = generate_data()
        output = model(src, tgt)
        loss = criterion(output.view(-1, 100), src.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    with torch.no_grad():
        test_src, _ = generate_data(1,5)
        test_tgt = torch.zeros(1,5,dtype=torch.long)
        pred = model(test_src, test_tgt).argmax(-1)
        print("Test:")
        print("Input:", test_src.numpy())
        print("Output:", pred.numpy())