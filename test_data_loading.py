from data_processing.data_reader import DataReader
from configuration.configuration import Configuration

# Load configuration
config = Configuration(mode="classifier")  # Use the classifier mode to test loading
data_reader = DataReader(
    caps_directories=config.caps_directories,
    info_data=config.info_data_files,
    diagnoses_info=config.diagnoses_info,
    quality_check=config.quality_check,
    valid_dataset_names=config.valid_dataset_names,
    info_data_cols=config.col_names
)

# Print sample data to confirm loading
print(data_reader.data.head())  # Check the data is loaded as expected
