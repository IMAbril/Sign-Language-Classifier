# CONFIG.PY
TRAINING_FILE = "input/sign_mnist_train_folds.csv"
TRAIN_FILE = "input/sign_mnist_train.csv"
TEST_FILE = "input/sign_mnist_test.csv"
MODEL_OUTPUT = "models/"
TEST_OUTPUT = "report/test_predictions.csv"
BEST_PARAMS = {'mlp': {'hidden_layer_sizes': (128,), 'PCA':35}}
