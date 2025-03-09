import os
import subprocess

# Optional: used for the Keras callback if using a TensorFlow/Keras model.
try:
    import tensorflow as tf
except ImportError:
    tf = None


class AutoSaveWeights:
    """
    This class provides helper methods to auto-save model weights after training.
    It includes a method to save weights using the model's save_weights functionality
    and then automatically commit and push the changes to Git.
    """

    def __init__(self, weights_path="model_weights.h5", git_commit_message="Update model weights"):
        self.weights_path = weights_path
        self.git_commit_message = git_commit_message

    def commit_weights(self):
        """
        Automatically stage, commit, and push the weights file to Git.
        Requires Git to be installed and configured with the necessary credentials.
        """
        try:
            # Stage the weights file
            subprocess.check_call(["git", "add", self.weights_path])
            # Commit the changes with the provided commit message
            subprocess.check_call(["git", "commit", "-m", self.git_commit_message])
            # Push to the remote repository
            subprocess.check_call(["git", "push"])
            print("Weights committed and pushed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error during git operations:", e)

    def save_and_commit(self, model):
        """
        Save the model weights and commit them via Git.
        This method assumes that 'model' has a 'save_weights' method.
        """
        try:
            model.save_weights(self.weights_path)
            print(f"Weights saved to {self.weights_path}.")
            self.commit_weights()
        except Exception as e:
            print("Error saving model weights:", e)


if tf:
    class AutoSaveWeightsCallback(tf.keras.callbacks.Callback):
        """
        A Keras Callback that automatically saves model weights and commits them in Git
        at the end of every epoch.
        """
        def __init__(self, weights_path="model_weights.h5", git_commit_message="Update model weights"):
            super().__init__()
            self.weights_path = weights_path
            self.git_commit_message = git_commit_message

        def on_epoch_end(self, epoch, logs=None):
            try:
                # Save weights at the end of the epoch
                self.model.save_weights(self.weights_path)
                print(f"Epoch {epoch+1}: Weights saved to {self.weights_path}.")
                # Stage the weights file for commit
                subprocess.check_call(["git", "add", self.weights_path])
                # Commit with a message that includes the epoch number
                subprocess.check_call(["git", "commit", "-m", f"{self.git_commit_message} after epoch {epoch+1}"])
                # Push to the remote repository
                subprocess.check_call(["git", "push"])
                print("Weights committed and pushed successfully.")
            except Exception as e:
                print("Error during auto-saving and committing weights:", e)


# USAGE EXAMPLES:
#
# For a generic model that supports a save_weights() method:
#
#     from auto_save_weights import AutoSaveWeights
#
#     # Assume your model is defined and trained somewhere.
#     auto_saver = AutoSaveWeights(weights_path="model_weights.h5")
#     # After training:
#     auto_saver.save_and_commit(model)
#
# For a Keras model, you can add the callback to the fit() method:
#
#     from auto_save_weights import AutoSaveWeightsCallback
#     model.fit(X_train, y_train, epochs=10, callbacks=[AutoSaveWeightsCallback()])
#
# Note:
# - The auto_commit feature uses subprocess calls to run Git commands.
#   Your environment must be configured with the appropriate Git credentials
#   (for example, via SSH keys or a credential helper) for these commands to succeed.
# - Adjust the weights_path and git_commit_message as needed.
