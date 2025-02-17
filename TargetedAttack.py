import numpy as np
import matplotlib.pyplot as plt
import pickle

def generate_image_for_class(model, target_class, alpha=0.1, learning_rate=0.01, max_iters=1000):
    """
    Generates a random image that will be classified as `target_class` by the neural network.

    Parameters:
    model: The neural network model (loaded from a trained file)
    target_class: The target class (integer) the image should be classified as (0-9 for MNIST)
    alpha: The range for initializing pixel values from Uniform(-alpha, alpha)
    learning_rate: The step size for gradient updates
    max_iters: The maximum number of iterations to run gradient descent

    Returns:
    x_adv: The adversarial image generated
    """
    # Step 1: Generate a random image x_0
    x_adv = np.random.uniform(-alpha, alpha, (784,))  # Random initialization

    for i in range(max_iters):
        # Compute the gradient of loss w.r.t x_adv
        grad_x = model.grad_wrt_input(x_adv, np.array([target_class]))

        # Ensure grad_x is the correct shape (flattened 784 pixels)
        if grad_x.shape[0] != 784:
            grad_x = grad_x[0]  # Extract the first image gradient if batch size is incorrect

        # Update x_adv using gradient descent
        x_adv -= learning_rate * grad_x

        # Get the predicted class of the current adversarial image
        predicted_class = np.argmax(model.forward(x_adv))

        # If the model classifies it as the target class, stop
        if predicted_class == target_class:
            print(f"Adversarial image successfully classified as {target_class} in {i + 1} iterations.")
            break

    return x_adv

def show_image(filename):
    """Load and display the adversarial image."""
    img = np.load(filename).reshape(28, 28)  # Reshape to 28x28
    plt.imshow(img, cmap="gray")
    plt.title(f"Adversarial Example from {filename}")
    plt.axis("off")  # Hide axis for better visualization
    plt.show()

def main():
    # Load the trained model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Generate adversarial images for each digit class (0-9)
    for c in range(10):
        x_adv = generate_image_for_class(model, target_class=c)

        # Save the generated image
        filename = f"targeted_random_img_class_{c}.npy"
        np.save(filename, x_adv)  # Save in NumPy format
        print(f"Saved adversarial image as {filename}")

        # Show the generated adversarial image
        show_image(filename)

# Execute main function
if __name__ == "__main__":
    main()



