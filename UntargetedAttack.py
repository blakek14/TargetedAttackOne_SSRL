import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


def fgsm(x_test, y_test, model, eps=0.05, max_attempts=10):
    """
    Performs an untargeted attack using Fast Gradient Sign Method (FGSM) until misclassification occurs.

    Parameters:
    x_test: Input image (flattened 784 pixels).
    y_test: True label of the image.
    model: Pre-trained neural network.
    eps: Perturbation magnitude.
    max_attempts: Maximum number of times to attempt misclassification.

    Returns:
    x_adv: Adversarial image that misclassifies the input or the closest attempt if unsuccessful.
    attempts: Number of iterations used.
    """

    x_adv = np.copy(x_test)
    attempts = 0

    while attempts < max_attempts:
        # Compute the gradient of loss w.r.t input x
        grad_x = model.grad_wrt_input(x_adv, np.array([y_test]))  # Ensure correct shape

        # Apply FGSM perturbation
        x_adv = x_adv + eps * np.sign(grad_x)  # Add sign of gradient times epsilon
        x_adv = np.clip(x_adv, 0, 1)  # Ensure pixel values remain valid (0-1 range)

        # Check if misclassification has occurred
        pred = np.argmax(model.forward(x_adv))
        if pred != y_test:
            return x_adv, attempts + 1  # Return misclassified image

        attempts += 1

    return x_adv, max_attempts  # Return last attempt if unsuccessful


def visualize_example(x, title):
    """Displays an image"""
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    # Load MNIST test dataset
    with open("mnist.pkl", "rb") as f:
        mnist = pickle.load(f)

    x_test = mnist["test_images"]  # Load test images
    y_test = mnist["test_labels"]  # Load test labels

    # Randomly select an image from the dataset
    random_index = random.randint(0, len(x_test) - 1)
    original_image = x_test[random_index]
    original_label = y_test[random_index]

    # Load pre-trained model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Get model's original prediction
    original_pred = np.argmax(model.forward(original_image))
    print(f"Original Image: True Label = {original_label}, Model Prediction = {original_pred}")

    # Generate adversarial image using FGSM with repeated attempts
    x_adv, attempts = fgsm(original_image, original_label, model)

    # Get model's prediction on adversarial image
    adversarial_pred = np.argmax(model.forward(x_adv))
    print(f"Adversarial Image: Model Prediction = {adversarial_pred}")
    print(f"Number of FGSM iterations: {attempts}")

    # Save the generated image
    np.save(f"FGSM_untargeted_{random_index}.npy", x_adv)

    # Visualize original vs adversarial image
    visualize_example(original_image, f"Original Image (Predicted: {original_pred})")
    visualize_example(x_adv, f"Adversarial Image (Predicted: {adversarial_pred})")

    # Check if attack was successful
    if adversarial_pred != original_label:
        print("Success")
    else:
        print("Fail")


if __name__ == "__main__":
    main()

