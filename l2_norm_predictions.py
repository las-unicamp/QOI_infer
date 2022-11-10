from typing import Literal
import numpy as np
import torch
from torchmetrics import MeanSquaredError


INDEX_OF_SELECTED_CASES = [1, 5, 8, 10, 13, 17, 18, 19, 20]
INDEX_OF_SELECTED_CASES = [3, 15]
INDEX_OF_DATASETS = [1, 2, 3, 4, 5]

NUMBER_OF_MEASUREMENTS_SUCTION_SIDE = 300

DataType = Literal["prediction", "target"]


def read_pressure_distribution(filepath: str, data_type: DataType) -> torch.Tensor:
    """
    Read and return either the predicted or true pressure distribution of the
    entire simulation.

    Parameters:
        filepath (str): path to the npy file with the values of pressure
            distribution of a given simulation.

    Returns:
        predictions (torch.Tensor)
    """
    available_dtypes = {
        "prediction": np.float32,
        "target": np.float64,
    }
    dtype = available_dtypes[data_type]

    pressure_distribution = np.fromfile(filepath, dtype=dtype).reshape(
        [-1, NUMBER_OF_MEASUREMENTS_SUCTION_SIDE]
    )

    return torch.from_numpy(pressure_distribution)


def evaluate_mse_for_each_snapshot(predictions, targets):
    assert len(predictions) == len(targets)

    number_of_snapshots = len(predictions)

    mean_squared_error = torch.zeros(number_of_snapshots)

    for i in range(number_of_snapshots):
        mean_squared_error_fn = MeanSquaredError()
        mean_squared_error[i] = mean_squared_error_fn(predictions[i], targets[i])

    return mean_squared_error


def main():
    for case in INDEX_OF_SELECTED_CASES:
        target_path = f"./outputs/targets_{case}_Cp_distribution.npy"

        targets = read_pressure_distribution(target_path, "target")

        for dataset in INDEX_OF_DATASETS:
            pred_path = (
                f"./outputs/predictions_DS{dataset}_{case}_Cp_Cp_distribution.npy"
            )

            predictions = read_pressure_distribution(pred_path, "prediction")

            mean_squared_error = evaluate_mse_for_each_snapshot(predictions, targets)

            mean_squared_error.cpu().detach().numpy().tofile(
                f"mse_DS{dataset}_{case}.npy"
            )


if __name__ == "__main__":
    main()
