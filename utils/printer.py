def printing_model(model, model_name):
    tot_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-------------------------------------------")
    print(f"{model_name} uploaded correctly!")
    print(f"Total parameters: {tot_params:,}")
    print(f"Trainable parameters: {train_params:,}")
    print(f"Non-trainable parameters: {tot_params - train_params:,}")
    print("-------------------------------------------")