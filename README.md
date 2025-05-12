# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0–9), often used for image processing tasks. The goal of this experiment is image denoising using autoencoders, a neural network designed to learn efficient representations. By introducing noise to images, the model is trained to reconstruct clean versions.

## DESIGN STEPS

### STEP 1:
Load MNIST dataset and convert to tensors.

### STEP 2:
Apply Gaussian noise to images for training.

### STEP 3:
Design encoder-decoder architecture for reconstruction.

### STEP 4:
Use MSE loss to measure reconstruction quality.

### STEP 5:
Train autoencoder using Adam optimizer efficiently.

### STEP 6:
Evaluate model on noisy and clean images.

### STEP 7:
Visualize results comparing original, noisy, denoised versions.

### STEP 8:
Improve performance by tuning hyperparameters carefully.

## PROGRAM
### Name: SETHUKKARASI C
### Register Number: 212223230201

```
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    # Include your code here
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")
```

## OUTPUT

### Model Summary

![model_summary](/summary.png)

### Original vs Noisy Vs Reconstructed Image

![image_visualization](/image.png)



## RESULT
A convolutional autoencoder for image denoising application is developed successfully.