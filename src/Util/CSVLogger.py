import pandas as pd
import os


class CSVLogger:
    def __init__(self, metrics, output_dir="."):
        self.counter = 1
        self.output_dir = os.path.join(output_dir, "log.csv")
        self.csv_file = pd.DataFrame(columns=["epoch"].extend(metrics))

    def _append(self, row={}):
        row["epoch"] = self.counter
        if False not in [True if i in row.keys() else False for i in self.csv_file.columns]:
            self.csv_file = self.csv_file.append(row, ignore_index=True)
            self.counter += 1
        else:
            print("Row could not add!\n", row)

    def _save(self):
        self.csv_file.to_csv(self.output_dir)

    def log(self, row):
        self._append(row)
        self._save()
