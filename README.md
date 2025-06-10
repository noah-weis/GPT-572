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

### What else should you know?

- If you would like to see the program in action, turn on development mode in config
- The whole project was developed on Windows, not many changes should be needed for Linux but I havn't tested.
- If you would like to actually train a fresh model, you'll need a long time and a nice GPU.
- I've left my best model in the models folder so you don't have to do training.
- That model is not the max performance of this repo, I didn't have enough training time.
