from data.preprocessing import load_and_prepare_data
from model.train import train_model
from model.evaluate import evaluate_model
from utils.plot_history import plot_history


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_and_prepare_data(
        "dataset/yes/*.*", "dataset/no/*.*"
    )
    model, history = train_model(X_train, y_train, X_val, y_val, class_weights)
    evaluate_model(model, X_test, y_test)
    plot_history(history)
    model.save("melanoma_model.h5")
    print("Modelis išsaugotas kaip 'melanoma_model.h5'")
