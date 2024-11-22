from torchvision import transforms
from dataset import FashionDataset
from helper_tester import ModelTesterMetrics
from model_base import FashionCNN
from helper_logger import DataLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import torch
import time
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_epochs = 40
batch_size = 16

if __name__ == "__main__":
    print("| Fashion Classification Training !")

    print("| Total Epoch :", total_epochs)
    print("| Batch Size  :", batch_size)
    print("| Device      :", device)

    logger = DataLogger("FashionClassification-FashionCNN")
    metrics = ModelTesterMetrics()

    # metrics.loss = torch.nn.BCEWithLogitsLoss()
    # metrics.activation = torch.nn.Softmax(1)

    # model = BasicMobileNet(7).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # training_augmentation = [
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # ]

    # Menambahkan class weights untuk mengatasi imbalance
    class_weights = torch.tensor(
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]).to(device)
    metrics.loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    metrics.activation = torch.nn.Softmax(1)

    model = FashionCNN(7).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.00005, weight_decay=0.0001)

    # Scheduler tanpa parameter verbose
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Tambahkan fungsi untuk mencetak learning rate
    def print_lr():
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

    # Simple Augmentation
    # training_augmentation = [
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomAffine(
    #         degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # ]

    # Augmentation Barbar
    training_augmentation = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(
            degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    ]

    # Menggunakan dataset yang sama dengan mode berbeda
    base_path = '../Dataset'
    validation_dataset = FashionDataset(base_path, mode='val')
    training_dataset = FashionDataset(
        base_path, mode='train', aug=training_augmentation)
    testing_dataset = FashionDataset(base_path, mode='test')

    validation_datasetloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True)
    training_datasetloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True)
    testing_datasetloader = DataLoader(
        testing_dataset, batch_size=1, shuffle=True)

    # logger = DataLogger("FashionClassification")
    # metrics = ModelTesterMetrics()

    # # Menambahkan class weights untuk mengatasi imbalance
    # class_weights = torch.tensor(
    #     [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]).to(device)
    # metrics.loss = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    # metrics.activation = torch.nn.Softmax(1)

    # model = BasicMobileNet(7).to(device)

    # # Menggunakan learning rate yang lebih kecil dan weight decay
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=0.00005, weight_decay=0.0001)

    # # Menambahkan learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=5,
    #     verbose=True
    # )

    # training_augmentation = [
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),  # Menambah rotasi
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomAffine(
    #         degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # ]

    total_start_time = time.time()  # waktu awal train

    # Training Evaluation Loop
    for current_epoch in range(total_epochs):

        epoch_start_time = time.time()  # waktu mulai epoch

        print("Epoch :", current_epoch)

        # Training Loop
        model.train()  # set the model to train
        metrics.reset()  # reset the metrics

        for (image, label) in tqdm(training_datasetloader, desc="Training :"):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = metrics.compute(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        training_mean_loss = metrics.average_loss()
        training_mean_accuracy = metrics.average_accuracy()

        # Evaluation Loop
        model.eval()    # set the model to evaluation
        metrics.reset()  # reset the metrics

        for (image, label) in tqdm(validation_datasetloader, desc="Testing  :"):

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            metrics.compute(output, label)

        evaluation_mean_loss = metrics.average_loss()
        evaluation_mean_accuracy = metrics.average_accuracy()

        # scheduler.step(evaluation_mean_accuracy)

        # Update learning rate dan cetak nilai baru
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(evaluation_mean_accuracy)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}")

        logger.append(
            current_epoch,
            training_mean_loss,
            training_mean_accuracy,
            evaluation_mean_loss,
            evaluation_mean_accuracy
        )

        epoch_end_time = time.time()  # waktu akhir epoch

        epoch_duration = epoch_end_time - epoch_start_time  # Durasi Epoch

        print(
            f"Epoch {current_epoch} completed in {epoch_duration:.2f} seconds.")

        if logger.current_epoch_is_best:
            print("> Latest Best Epoch :", logger.best_accuracy())
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictonary = {
                "model_state": model_state,
                "optimizer_state": optimizer_state
            }
            torch.save(
                state_dictonary,
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")

    total_end_time = time.time()  # waktu akhir train
    total_duration = total_end_time - total_start_time
    print(f"Training completed in {total_duration:.2f} seconds.")

    print("| Training Complete, Loading Best Checkpoint")

    # Load Model State
    state_dictonary = torch.load(
        logger.get_filepath("best_checkpoint.pth"),
        map_location=device
    )
    model.load_state_dict(state_dictonary['model_state'])
    model = model.to(device)

    # Testing System
    model.eval()    # set the model to evaluation
    metrics.reset()  # reset the metrics

    for (image, label) in tqdm(testing_datasetloader):

        image = image.to(device)
        label = label.to(device)

        output = model(image)
        metrics.compute(output, label)

    testing_mean_loss = metrics.average_loss()
    testing_mean_accuracy = metrics.average_accuracy()

    print("")
    logger.write_text(f"# Final Testing Loss     : {testing_mean_loss}")
    logger.write_text(f"# Final Testing Accuracy : {testing_mean_accuracy}")
    logger.write_text(f" Training Time           : {total_duration} Seconds")
    logger.write_text(f"# Report :")
    logger.write_text(metrics.report())
    logger.write_text(f"# Confusion Matrix :")
    logger.write_text(metrics.confusion())

    print("")
