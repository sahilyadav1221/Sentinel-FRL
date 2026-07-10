import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

# Dummy dataset
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model & optimizer
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Attach Opacus
privacy_engine = PrivacyEngine()

model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,   # ✅ must be a DataLoader
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

print("✅ Opacus working on:", "CUDA" if torch.cuda.is_available() else "CPU")
