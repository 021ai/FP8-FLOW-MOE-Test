import csv
from pathlib import Path


class BenchmarkCSVWriter:
    """Collect benchmark rows and write them to a CSV file."""

    def __init__(self, output_path, fieldnames):
        self.output_path = Path(output_path)
        self.fieldnames = list(fieldnames)
        self.rows = []

    @classmethod
    def for_script(cls, script_file, fieldnames, suffix='.csv'):
        script_path = Path(script_file).resolve()
        output_path = script_path.with_name(script_path.stem + suffix)
        return cls(output_path, fieldnames)

    def add_row(self, **row):
        unknown_fields = set(row) - set(self.fieldnames)
        if unknown_fields:
            raise ValueError(
                f'Unknown CSV fields: {sorted(unknown_fields)}. '
                f'Expected fields: {self.fieldnames}'
            )
        self.rows.append(row)

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.output_path