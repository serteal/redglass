from torch.utils.data import DataLoader

from redglass import data, detectors, models, scripts

# ===== MODEL =====
model = models.HuggingFaceModel(name_or_path="meta-llama/Llama3", config=...)
# model = models.OpenAIModel(name_or_path="gpt-3.5-turbo", config=...)

model.forward("hello world")
# model.forward({"role": "user", "content": "hello world"})
# model.forward(list of tokens)
# model.forward(continuous embeddings)

model.generate("hello world")
# model.generate(list of tokens)
# model.generate(continuous embeddings)

# ===== DETECTOR =====
detector = detectors.linear.CosineSimilarity(config=...)
# detector = detectors.mlp.MLP()
# detector = detectors.sae.FeatureScore()
# detector = detectors.anomaly.VAE()

scripts.train_detector(
    detector=detector,
    model=model,
    train_loader=DataLoader(
        data.ContrastPairDataset(
            behavior_data=data.HuggingFaceDataset(url="advbench"),
            normal_data=data.HuggingFaceDataset(url="alpaca"),
        )
    ),
    val_loaders={
        "normal_data": DataLoader(data.HuggingFaceDataset(url="alpaca", train=False)),
        "behavior_data": DataLoader(data.HuggingFaceDataset(url="advbench", train=False)),
    },
    config=...,
)

scripts.eval_detector(detector=detector, model=model, val_dataloader=DataLoader())

# ====== ADVERSARIAL EXAMPLES ======

scripts.train_adversa
