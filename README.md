# Setup and Run Instructions

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model

The model can be run with different flags for different purposes:

```bash
python main.py              # Train the model with default settings
python main.py --tune      # Run hyperparameter tuning
python main.py --evaluate  # Evaluate model quality
python main.py --generate  # Generate text using a saved model
python main.py --clear     # Clear all saved models and tuning data
```

## Project Structure

- `main.py`: Main entry point. Handles command-line arguments for training, tuning, evaluation, and text generation.
- `gpt.py`: Defines the GPT language model architecture and logic.
- `trainer.py`: Contains training routines, hyperparameter tuning (Optuna), and text generation utilities.
- `evaluator.py`: Provides tools for evaluating generated text quality (perplexity, coherence, diversity).
- `data_loader.py`: Loads and preprocesses datasets, handles tokenization, and creates data splits/loaders.
- `config.py`: Central configuration for model, data, training, and generation settings.
- `requirements.txt`: Lists required Python dependencies.
- `study.db`: Database file used by Optuna for storing hyperparameter tuning results.
- `models/`: Directory for saving trained and tuned model files.

### What else should you know?

- If you would like to see the program in action, turn on development mode in config
- The whole project was developed on Windows, not many changes should be needed for Linux but I havn't tested.
- If you would like to actually train a fresh model, you'll need a long time and a nice GPU.
- The model is too big to be uploaded to GitHub, but the study.db and current best hyperparameters are uploaded.
- That model is not the max performance of this repo, it's still tuning.
