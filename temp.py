import torch  # Main PyTorch package for tensor operations and deep learning functionalities
import torch.nn as nn  # Submodule for defining neural network layers like nn.Linear, nn.Conv2d, etc.
import torch.optim as optim  # Optimizers such as Adam, SGD for model training
from torch.utils.data import Dataset, DataLoader, random_split  # Dataset: abstract class for data, DataLoader: batching, random_split: dataset partitioning
import numpy as np  # Numerical operations and array manipulation, often used for preprocessing and conversion
from pymongo import MongoClient  # Interface for connecting to MongoDB — used to retrieve or store PSF data and labels
from tqdm import tqdm  # Displays smart progress bars for training/validation loops and batch iterations
import copy  # Enables deep copying of model weights (useful for checkpointing the best model)
import matplotlib.pyplot as plt  # Plotting utility for visualizations like training/validation loss curves and PSF comparisons
import warnings  # Python’s standard warning control module — used here to suppress non-critical warnings
import torch.nn.functional as F  # Provides stateless versions of activation functions and loss functions (e.g., F.relu, F.mse_loss)
import os  # OS interface to create directories, manage file paths, and access environment-specific variables
import yaml  # Parses YAML configuration files into Python dictionaries (used if OmegaConf isn’t applied)
from omegaconf import OmegaConf  # More powerful configuration handler than `yaml` — supports dot-access and structured configs
from omegaconf import DictConfig, ListConfig  # Typed configuration containers supporting dot access and validation
import random  # Python’s built-in random module — used for global seeding and randomness control in splits or shuffling

warnings.filterwarnings('ignore')  # Silences all warning messages to keep console/log outputs clean during training

# Check if the dataset configuration file exists
if not os.path.exists('./dataset_config.yaml'):
    raise FileNotFoundError("Dataset config file not found.")

# Check if the model configuration file exists
if not os.path.exists('./model_config.yaml'):
    raise FileNotFoundError("Model config file not found.")

# Load dataset configuration from a YAML file into an OmegaConf object
dataset_config = OmegaConf.load('./dataset_config.yaml')  # Contains settings like dataset paths, preprocessing, seed, etc.

# Load model architecture and training configuration from another YAML file
model_config = OmegaConf.load('./model_config.yaml')  # Contains model details like layers, optimizer settings, and training hyperparameters

# Merge the dataset and model configs into a single config dictionary-like object
config = OmegaConf.merge(dataset_config, model_config)  # Allows unified access to all config parameters using dot notation

# Validate MongoDB configuration
if 'mongodb' not in config:
    raise KeyError("MongoDB configuration section is missing in the configuration file.")

if 'uri' not in config['mongodb']:
    raise KeyError("MongoDB URI is missing in the configuration file.")
if 'database' not in config['mongodb']:
    raise KeyError("MongoDB database name is missing in the configuration file.")
if 'collection' not in config['mongodb']:
    raise KeyError("MongoDB collection name is missing in the configuration file.")

if not isinstance(config['mongodb']['uri'], str):
    raise TypeError("MongoDB URI must be a string.")
if not isinstance(config['mongodb']['database'], str):
    raise TypeError("MongoDB database name must be a string.")
if not isinstance(config['mongodb']['collection'], str):
    raise TypeError("MongoDB collection name must be a string.")

# Attempt to connect to MongoDB and create index
try:
    client = MongoClient(config['mongodb']['uri'])  # Connect to MongoDB using the URI
    db = client[config['mongodb']['database']]  # Access the database
    full_collection = db[config['mongodb']['collection']]  # Access the collection
    full_collection.create_index("index", name="idx", unique=True)  # Ensure unique index field
except Exception as e:
    raise ConnectionError(f"Failed to connect to MongoDB or create index: {e}")

# Validate and set seeds for reproducibility
if "dataset" not in config or "seed" not in config["dataset"]:
    raise KeyError("Seed value is missing from the configuration under 'dataset.seed'.")

seed = config["dataset"]["seed"]

if not isinstance(seed, int):
    raise TypeError(f"Seed must be an integer, got {type(seed)} instead.")

# Set seeds across all relevant libraries
random.seed(seed)  # Python's built-in random
np.random.seed(seed)  # NumPy random
torch.manual_seed(seed)  # PyTorch CPU RNG
torch.cuda.manual_seed_all(seed)  # PyTorch CUDA RNGs (all GPUs)
torch.backends.cudnn.deterministic = True  # Force deterministic behavior in cuDNN
torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility

def validate_mirrors(config):
    """
    Validate the mirror configurations in the config file.

    This function checks that all required mirror segments and their parameters are properly defined.
    It verifies that each segment has the necessary piston, tip, and tilt parameters and that 
    they are correctly formatted.

    Parameters:
    -----------
    config : dict
        Configuration dictionary loaded from the YAML file
        
    Raises:
    -------
    ValueError
        If any mirror configuration is invalid or missing
    """
    # Validate primary mirror configuration
    if 'primary_mirror' not in config:
        raise ValueError("Missing 'primary_mirror' section in config.")

    # Check if 'segments' exists in the primary_mirror section
    if 'segments' not in config['primary_mirror']:
        raise ValueError("Missing 'segments' key in 'primary_mirror' section.")

    segments = config['primary_mirror']['segments']
    # Use isinstance with (list, ListConfig) to handle both Python lists and OmegaConf ListConfig objects
    if not isinstance(segments, (list, ListConfig)):
        raise ValueError("'segments' under 'primary_mirror' must be a list.")

    found_ids = set()
    missing_params_by_id = {}

    # Validate each segment in the primary mirror
    for i, segment in enumerate(segments):
        # Validate segment has an ID
        if 'id' not in segment:
            raise ValueError(f"Segment at index {i} is missing 'id' key.")
        
        seg_id = segment['id']
        if not isinstance(seg_id, int):
            raise ValueError(f"Segment at index {i} has a non-integer 'id': {seg_id}")
        
        # Check for duplicate IDs
        if seg_id in found_ids:
            raise ValueError(f"Duplicate segment ID found: {seg_id}")
        
        found_ids.add(seg_id)

        # Check for required parameters (piston, tip, tilt)
        missing_keys = []
        for key in ['piston', 'tip', 'tilt']:
            if key not in segment:
                missing_keys.append(key)
                continue

            # Validate parameter format (must be a dict with either 'static', 'range', or 'untrained_range')
            param_dict = segment[key]
            if not isinstance(param_dict, (dict, DictConfig)):
                raise ValueError(f"{key} for segment {seg_id} must be a dictionary.")
            
            # Check that exactly one of the allowed keys is present
            allowed_keys = {'static', 'range', 'untrained_range'}
            param_keys = set(param_dict.keys())
            if not param_keys.issubset(allowed_keys):
                invalid_keys = param_keys - allowed_keys
                raise ValueError(f"{key} for segment {seg_id} contains invalid keys: {invalid_keys}. "
                               f"Must use only: {allowed_keys}.")
            
            if len(param_dict) != 1:
                raise ValueError(f"{key} for segment {seg_id} must contain exactly one of {allowed_keys}.")
            
            # Get the parameter type (static or range)
            inner_key = next(iter(param_dict))
            
            # Validate the value based on parameter type
            value = param_dict[inner_key]
            if inner_key == 'static':
                # Static values must be floats
                if not isinstance(value, (int, float)):  # Allow both int and float for flexibility
                    raise ValueError(f"{key} for segment {seg_id} must be a number under 'static', but got {type(value).__name__}.")
            else:
                # Range values must be a list of exactly 2 floats
                if not isinstance(value, (list, ListConfig)):
                    raise ValueError(f"{key} for segment {seg_id} under '{inner_key}' must be a list, but got {type(value).__name__}.")
                
                if len(value) != 2:
                    raise ValueError(f"{key} for segment {seg_id} under '{inner_key}' must be a list of exactly 2 values.")
                
                if not all(isinstance(v, (int, float)) for v in value):
                    non_numeric = [i for i, v in enumerate(value) if not isinstance(v, (int, float))]
                    raise ValueError(f"{key} for segment {seg_id} under '{inner_key}' contains non-numeric values at positions {non_numeric}.")

        # Track any missing parameters for this segment
        if missing_keys:
            missing_params_by_id[seg_id] = missing_keys

    # Ensure all required segments (1-6) are present
    required_ids = set(range(1, 7))
    missing_ids = sorted(list(required_ids - found_ids))
    if missing_ids:
        raise ValueError(f"Missing segments with IDs: {missing_ids}")

    # Report any segments with missing parameters
    if missing_params_by_id:
        msg = "Some segments are missing required parameters:\n"
        for sid, keys in missing_params_by_id.items():
            msg += f"  Segment ID {sid}: missing {', '.join(keys)}\n"
        raise ValueError(msg)

    # Validate secondary mirror configuration
    if 'secondary_mirror' not in config:
        raise ValueError("Missing 'secondary_mirror' section in config.")

    secondary_params = config['secondary_mirror']

    # Check that secondary mirror has all required parameters
    for param in ['piston', 'tip', 'tilt']:
        if param not in secondary_params:
            raise ValueError(f"Missing '{param}' parameter in 'secondary_mirror'.")

        # Validate parameter format
        param_dict = secondary_params[param]
        if not isinstance(param_dict, (dict, DictConfig)):
            raise ValueError(f"{param} for secondary mirror must be a dictionary, but got {type(param_dict).__name__}.")
        
        # Check that exactly one of the allowed keys is present
        allowed_keys = {'static', 'range', 'untrained_range'}
        param_keys = set(param_dict.keys())
        if not param_keys.issubset(allowed_keys):
            invalid_keys = param_keys - allowed_keys
            raise ValueError(f"{param} for secondary mirror contains invalid keys: {invalid_keys}. "
                           f"Must use only: {allowed_keys}.")
        
        if len(param_dict) != 1:
            raise ValueError(f"{param} for secondary mirror must contain exactly one of {allowed_keys}.")
        
        # Get the parameter type
        inner_key = next(iter(param_dict))
        
        # Validate the value based on parameter type
        value = param_dict[inner_key]
        if inner_key == 'static':
            # Static values must be numbers (int or float)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{param} for secondary mirror must be a number under 'static', but got {type(value).__name__}.")
        else:
            # Range values must be a list of two numbers
            if not isinstance(value, (list, ListConfig)):
                raise ValueError(f"{param} for secondary mirror under '{inner_key}' must be a list, but got {type(value).__name__}.")
            
            if len(value) != 2:
                raise ValueError(f"{param} for secondary mirror under '{inner_key}' must be a list of exactly 2 values.")
            
            if not all(isinstance(v, (int, float)) for v in value):
                non_numeric = [i for i, v in enumerate(value) if not isinstance(v, (int, float))]
                raise ValueError(f"{param} for secondary mirror under '{inner_key}' contains non-numeric values at positions {non_numeric}.")

    return True  # Return True if validation passes

validate_mirrors(config);

def extract_parameters(config):
    # Lists to categorize parameters based on training roles
    to_predict = []           # Parameters with defined "range" — included in training
    untrained_predict = []    # Parameters with "untrained_range" — excluded from training, used for testing/generalization
    not_to_predict = []       # Parameters marked "static" — constant values, not predicted

    # Iterate through each segment in the primary mirror
    for segment in config["primary_mirror"]["segments"]:
        seg_id = segment["id"]  # Segment identifier (e.g., 1, 2, 3...)

        for param in ["piston", "tip", "tilt"]:  # Loop through each degree of freedom (DoF)
            param_config = segment[param]  # Access configuration for this DoF
            param_name = f"primary_{seg_id}_{param}"  # Create parameter name, e.g., "primary_1_piston"

            # Classify the parameter into the appropriate category
            if "range" in param_config:
                to_predict.append(param_name)  # Included in model training
            elif "untrained_range" in param_config:
                untrained_predict.append(param_name)  # Held out for generalization testing
            elif "static" in param_config:
                not_to_predict.append(param_name)  # Fixed parameter, excluded from prediction

    # Process secondary mirror parameters in the same way
    for param in ["piston", "tip", "tilt"]:
        param_config = config["secondary_mirror"][param]  # Access config for each DoF
        param_name = f"secondary_{param}"  # e.g., "secondary_tip"

        # Classify the secondary mirror parameter
        if "range" in param_config:
            to_predict.append(param_name)
        elif "untrained_range" in param_config:
            untrained_predict.append(param_name)
        elif "static" in param_config:
            not_to_predict.append(param_name)

    # Return all categorized parameter names
    return to_predict, untrained_predict, not_to_predict

to_predict, untrained_predict, not_to_predict = extract_parameters(config)

# Helper function to flatten a nested dictionary into a flat dictionary
def flatten_dict(d, parent_key='', sep='_'):
    items = []  # Initialize an empty list to store the flattened key-value pairs
    
    for k, v in d.items():  # Iterate through each key-value pair in the dictionary
        # If a parent_key exists, join it with the current key using the separator
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Construct the flattened key
        
        if isinstance(v, dict):  # Check if the value is a dictionary
            # Recursively flatten the nested dictionary and extend the result to the items list
            items.extend(flatten_dict(v, new_key, sep=sep).items())  # Flatten the nested dict and add to the list
        else:
            # If the value is not a dictionary, add the key-value pair directly to the list
            items.append((new_key, v))  # Add the key-value pair to the flattened list
    
    return dict(items)  # Return the flattened dictionary as a dict

class DBImageDataset(Dataset):
    def __init__(self, config, shuffle=True):
        """
        Initialize the dataset with merged config and shuffle flag.

        :param config: Merged configuration object
        :param shuffle: Flag to control shuffling of dataset
        """
        self.config = config  # Store the dataset configuration
        self.mongo_collection = full_collection  # MongoDB collection for fetching documents (must be pre-initialized)
        self.target_keys = to_predict  # Target parameters that the model should predict
        self.shuffle = shuffle  # Flag for whether to shuffle the dataset on initialization

        # --- Defocus Validation and Handling ---
        raw_defocus = self.config["dataset"]["use_defocus"]  # Raw value from config (can be int or float)

        # Ensure that 'use_defocus' is a numeric type (int or float)
        if not isinstance(raw_defocus, (int, float)):
            raise TypeError(f"'use_defocus' must be an int or float, got {type(raw_defocus)}.")
        
        # Convert to boolean: 0 means no defocus used, non-zero means defocus applied
        self.use_defocused = bool(raw_defocus)

        # --- Transform Validation ---
        self.transform = self.config["dataset"]["transform"]  # Type of transform to apply on PSFs
        allowed_transforms = {"linear", "sqrt", "log"}  # Only these three are allowed
        
        # Validate that transform type is one of the supported options
        if self.transform not in allowed_transforms:
            raise ValueError(f"Unsupported transform '{self.transform}'. Must be one of {allowed_transforms}.")

        # --- Load Dataset Indexes ---
        # Fetch all document indexes from the MongoDB collection (used to retrieve samples efficiently)
        self.doc_indexes = list(
            self.mongo_collection.find({}, {"index": 1, "_id": 0}).sort("index", 1)
        )
        
        # Raise an error if no data was found
        if not self.doc_indexes:
            raise ValueError("No documents found in the collection. Ensure each document has an 'index' field.")

        self.total_docs = len(self.doc_indexes)  # Store the number of total samples available

        # Shuffle the list of indexes if requested
        if self.shuffle:
            np.random.shuffle(self.doc_indexes)

    def __len__(self):
        # Returns the total number of samples in the dataset
        return self.total_docs

    def __getitem__(self, idx):
        """
        Fetch and process a data item by its index.

        :param idx: Index of the sample
        :return: A tuple of image tensor, (optional magnitude tensor), and target tensor
        """
        # Get the unique document index for this sample
        unique_index = self.doc_indexes[idx]["index"]
        
        # Retrieve the full document using its unique index
        doc = self.mongo_collection.find_one({"index": unique_index})
        
        # Flatten the document into a single-level dictionary for easier access
        flat = flatten_dict(doc)

        # --- Extract Target Parameters ---
        targets = [flat[key] for key in self.target_keys]  # Extract target values from the flattened dict
        target_tensor = torch.tensor(targets, dtype=torch.float32)  # Convert to PyTorch tensor

        # --- Load PSF(s) ---
        nominal = np.array(doc["psf_nominal_array"], dtype=np.float32)  # Load nominal PSF

        # Optionally load defocused PSF if defocus is enabled
        if self.use_defocused:
            defocused = np.array(doc["psf_defocused_array"], dtype=np.float32)

        # --- Apply Transformation ---
        if self.transform == "sqrt":
            nominal = np.sqrt(nominal)  # Apply square root transform
            if self.use_defocused:
                defocused = np.sqrt(defocused)
        elif self.transform == "log":
            nominal = np.log(nominal + 1e-6)  # Apply log transform (with epsilon for stability)
            if self.use_defocused:
                defocused = np.log(defocused + 1e-6)
        # No transform applied for "linear"

        # --- Stack PSFs ---
        image_channels = [nominal]  # Start with nominal PSF as the first channel
        if self.use_defocused:
            image_channels.append(defocused)  # Add defocused PSF if applicable

        image_np = np.stack(image_channels, axis=0)  # Stack along channel axis (C, H, W)
        image_tensor = torch.from_numpy(image_np)  # Convert to PyTorch tensor

        # --- Optional Phase-Magnitude Features ---
        if self.config["dataset"]["use_phase_mag"]:
            row_col_features = []  # Store 1D features from frequency domain
            psfs = [nominal]  # Start with nominal PSF
            if self.use_defocused:
                psfs.append(defocused)  # Include defocused if applicable

            for img in psfs:
                ifft_result = np.fft.ifft2(img)  # Perform inverse FFT
                magnitude = np.abs(ifft_result)  # Get magnitude spectrum
                h, w = magnitude.shape
                mid_row = magnitude[h // 2, :]  # Extract middle row
                mid_col = magnitude[:, w // 2]  # Extract middle column
                row_col_features.extend([mid_row, mid_col])  # Append both

            # Concatenate and convert to tensor
            magnitude_tensors = torch.tensor(np.concatenate(row_col_features), dtype=torch.float32)
            return image_tensor, magnitude_tensors, target_tensor  # Return all three tensors

        # Return image and target tensors (no magnitude features)
        return image_tensor, target_tensor

# --- Validate Config Values ---

# Validate that 'seed' is present and is an integer
if "seed" not in config["dataset"]:
    raise KeyError("'seed' must be defined in config['dataset'].")
if not isinstance(config["dataset"]["seed"], int):
    raise TypeError(f"'seed' must be an integer, got {type(config['dataset']['seed'])}.")

# Validate that 'batch_size' is present and is an integer
if "batch_size" not in config["training"]:
    raise KeyError("'batch_size' must be defined in config['training'].")
if not isinstance(config["training"]["batch_size"], int):
    raise TypeError(f"'batch_size' must be an integer, got {type(config['training']['batch_size'])}.")

# --- Load the Full Dataset ---

# Load the full dataset from a single collection
full_dataset = DBImageDataset(
    config=config,  # Pass the configuration containing all dataset-related settings
    shuffle=False   # Disable shuffling here; shuffling will be handled at DataLoader level or after splitting
)

# --- Dataset Splitting ---

# Define split ratios for the dataset
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# Calculate total number of samples
total_size = len(full_dataset)

# Compute sizes for each split
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # Ensure all samples are used

# Extract and use the seed from config
split_seed = config["dataset"]["seed"]

# Perform reproducible random split using the validated seed
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(split_seed)
)

# --- Create DataLoaders ---

# Extract validated batch size
batch_size = config["training"]["batch_size"]

# Create DataLoader for each dataset split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Shuffle training data
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)      # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # No shuffle for testing

class ResNetRegression(nn.Module):
    def __init__(self, config, num_outputs):
        super(ResNetRegression, self).__init__()
        self.cfg = config["model"]
        self.use_phase_mag = config["dataset"]["use_phase_mag"]  # Flag to use phase-magnitude ANN branch
        self.merge_strategy = self.cfg["add_magnitude_ann"]["merge_strategy"] if self.use_phase_mag else None

        # --- Input Configuration ---
        zemax_sampling = config["zemax"]["sampling"]
        self.image_size = 64 * zemax_sampling  # Assumes 64x64 base PSF size scaled by sampling
        self.input_channels = 2 if config["dataset"]["use_defocus"] else 1  # 1 for nominal, 2 if defocus is used
        self.output_dim = num_outputs  # Number of output regression parameters

        # --- Convolutional Backbone ---
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels
        for layer_cfg in self.cfg["conv_layers"]:
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            padding = kernel_size // 2  # Keep spatial dimensions consistent
            activation = layer_cfg.get("activation", True)  # Default to using ReLU

            block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)]
            if activation:
                block.append(nn.ReLU())
            if layer_cfg["pooling"]:
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample spatial size by 2

            self.conv_layers.append(nn.Sequential(*block))
            in_channels = out_channels  # Update for next layer

        # --- Skip Connections ---
        self.skip_connections = self.cfg["skip_connections"]
        self.skip_ops = nn.ModuleDict()
        for skip in self.skip_connections:
            from_layer = skip["from_layer"]
            to_layer = skip["to_layer"]
            skip_type = skip["type"]
            in_ch = self.cfg["conv_layers"][from_layer]["out_channels"]
            out_ch = self.cfg["conv_layers"][to_layer]["out_channels"]

            if skip_type == "conv+pool":
                # Align dimensions using downsampling and 1x1 convolution
                self.skip_ops[f"{from_layer}->{to_layer}"] = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                )
            elif skip_type == "add":
                self.skip_ops[f"{from_layer}->{to_layer}"] = nn.Identity()

        # --- Flatten Layer Output ---
        pool_count = sum(1 for l in self.cfg["conv_layers"] if l["pooling"])  # Count how many times we halved the size
        spatial_dim = self.image_size // (2 ** pool_count)  # Final spatial dimension after pooling
        final_out_channels = self.cfg["conv_layers"][-1]["out_channels"]
        flatten_dim = final_out_channels * spatial_dim * spatial_dim

        # --- Fully Connected Layers for CNN Path ---
        self.fc_layers = nn.ModuleList()
        in_dim = flatten_dim
        for dim in self.cfg["fc_layers"]:
            self.fc_layers.append(nn.Linear(in_dim, dim))
            in_dim = dim
        self.final_cnn_out_dim = in_dim  # Save final output size of CNN for merging

        # --- ANN Branch for Phase Magnitude Features ---
        if self.use_phase_mag:
            ann_hidden = self.cfg["add_magnitude_ann"]["hidden_layers"]
            self.ann_layers = nn.ModuleList()
            
            # Each PSF contributes row and column, so 2 * image_size; defocused adds another 2
            ann_in_dim = self.image_size * 2
            if config["dataset"]["use_defocus"]:
                ann_in_dim *= 2

            # Create fully connected layers for ANN
            for dim in ann_hidden:
                self.ann_layers.append(nn.Linear(ann_in_dim, dim))
                ann_in_dim = dim
            self.ann_out_dim = ann_in_dim  # Final ANN output dimension

            # Determine final dimension after merging CNN and ANN
            if self.merge_strategy == "concat":
                merged_dim = self.final_cnn_out_dim + self.ann_out_dim
            elif self.merge_strategy == "add":
                assert self.final_cnn_out_dim == self.ann_out_dim, "Merge='add' requires ANN and CNN dims to match"
                merged_dim = self.final_cnn_out_dim

            self.fc_out = nn.Linear(merged_dim, self.output_dim)
        else:
            # If ANN is not used, project CNN output directly
            self.fc_out = nn.Linear(self.final_cnn_out_dim, self.output_dim)

    def forward(self, *inputs):
        # Support for optional second input (magnitude features)
        if self.use_phase_mag:
            x, magnitude_tensors = inputs
        else:
            x = inputs[0]

        # --- Forward Pass through Conv Layers with Skip Connections ---
        outputs = []
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            outputs.append(x)

            # Apply skip connection if defined for this layer
            for skip in self.skip_connections:
                if skip["to_layer"] == i:
                    from_layer = skip["from_layer"]
                    key = f"{from_layer}->{i}"
                    skip_out = self.skip_ops[key](outputs[from_layer])
                    x = x + skip_out  # Residual addition
                    outputs[i] = x

        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Flatten tensor for FC layers

        # Pass through CNN's FC layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))

        # --- Optional ANN Forward Pass and Merge ---
        if self.use_phase_mag:
            ann = magnitude_tensors.view(magnitude_tensors.size(0), -1)
            for layer in self.ann_layers:
                ann = F.relu(layer(ann))

            if self.merge_strategy == "concat":
                x = torch.cat([x, ann], dim=1)
            elif self.merge_strategy == "add":
                x = x + ann

        # Final regression output
        return self.fc_out(x)

class CustomRMSELoss(nn.Module):
    def __init__(self):
        super(CustomRMSELoss, self).__init__()

    def forward(self, predictions, targets):
        # predictions: Tensor of shape (B, N), where B = batch size, N = number of outputs
        # targets:     Tensor of shape (B, N)

        # Compute MSE per sample (along the output dimension)
        row_mse = torch.mean((predictions - targets) ** 2, dim=1)  # Shape: (B,)

        # Take sqrt to get RMSE per sample
        row_rmse = torch.sqrt(row_mse)  # Shape: (B,)

        # Average over batch to get final scalar loss
        return torch.mean(row_rmse)

def plot_training_history(train_losses, val_losses, test_loss, num_epochs, config):
    # Extract model name and base path from config
    model_name = config["model"]["name"]
    model_base_path = config["model"]["base_path"]
    
    # Construct the full path where the plot will be saved
    save_path = os.path.join(model_base_path, f"loss_curve_{model_name}.png")
    
    # Define epoch indices for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a new figure for the loss curves
    plt.figure(figsize=(10, 5))
    
    # Plot training loss (solid line with circles)
    plt.plot(epochs, train_losses, label="Training Loss", marker='o', linestyle='-')
    
    # Plot validation loss (dashed line with squares)
    plt.plot(epochs, val_losses, label="Validation Loss", marker='s', linestyle='--')
    
    # Plot a horizontal line for the test loss
    plt.axhline(y=test_loss, color='r', linestyle='-.', label=f"Test Loss: {test_loss:.4f}")

    # Set axis labels and plot title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training, Validation & Test Loss — {model_name}")
    
    # Show legend and grid for better readability
    plt.legend()
    plt.grid(True)

    # Make sure the output directory exists before saving the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot to the specified file path
    plt.savefig(save_path)
    print(f"Training plot saved to: {save_path}")

    # Display the plot inline
    plt.show()

def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, device, config):
    """
    Train and evaluate the model across multiple epochs, including saving the best model based on validation loss.
    
    :param model: The neural network model to be trained.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param test_loader: DataLoader for the test dataset.
    :param num_epochs: Number of epochs to train the model.
    :param optimizer: The optimizer used to update model weights.
    :param criterion: The loss function used to calculate the loss.
    :param device: The device to run the model (e.g., "cuda" or "cpu").
    :param config: Configuration dictionary containing various parameters (e.g., dataset settings, model settings).
    
    :return: The trained model with the best validation performance.
    """
    
    # Initialize variables to track the best validation loss and model weights
    best_val_loss = float('inf')  # Start with a high initial value for validation loss
    best_model_wts = None  # Placeholder for storing the best model weights

    # Lists to track training and validation losses, as well as average losses
    train_losses = []
    val_losses = []
    avg_losses = []

    # Retrieve model name, save path, and dataset-specific settings from the config
    model_name = config["model"]["name"]
    model_base_path = config["model"]["base_path"]
    use_phase_mag = config["dataset"]["use_phase_mag"]

    # Create the model save directory if it doesn't exist
    os.makedirs(model_base_path, exist_ok=True)

    # Training loop across all epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0  # Initialize the total training loss for this epoch
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=True)  # Progress bar

        # Iterate over the batches in the training set
        for batch in train_bar:
            if use_phase_mag:
                # If phase magnitude is used, unpack both image and magnitude features
                inputs, magnitude_features, targets = batch
                magnitude_features = magnitude_features.to(device)  # Move magnitude features to device
                model_input = (inputs.to(device), magnitude_features)  # Create input tuple for the model
            else:
                # Otherwise, just unpack the image and target
                inputs, targets = batch
                model_input = (inputs.to(device),)  # Input tuple only contains the image

            targets = targets.to(device)  # Move targets to device

            optimizer.zero_grad()  # Zero out the gradients from the previous step
            outputs = model(*model_input)  # Forward pass through the model
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backpropagate the loss to compute gradients
            optimizer.step()  # Update the model weights using the optimizer

            # Accumulate the loss for the entire batch
            train_loss += loss.item() * inputs.size(0)  # Multiply by batch size for proper loss averaging
            train_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update progress bar with current loss

        # Calculate the average training loss for this epoch
        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)  # Store the training loss for plotting

        # Validation phase (no gradient computation)
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0  # Initialize the validation loss
        with torch.no_grad():  # No need to track gradients during validation
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=True)  # Progress bar
            # Iterate over validation batches
            for batch in val_bar:
                if use_phase_mag:
                    # Unpack inputs, magnitude features, and targets if using phase magnitude
                    inputs, magnitude_features, targets = batch
                    magnitude_features = magnitude_features.to(device)
                    model_input = (inputs.to(device), magnitude_features)
                else:
                    # Otherwise, unpack image and targets
                    inputs, targets = batch
                    model_input = (inputs.to(device),)

                targets = targets.to(device)  # Move targets to device

                # Forward pass without gradient computation
                outputs = model(*model_input)
                loss = criterion(outputs, targets)  # Compute the validation loss
                val_loss += loss.item() * inputs.size(0)  # Accumulate the loss for the batch
                val_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update progress bar with current loss

        # Calculate the average validation loss for this epoch
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)  # Store the validation loss for plotting

        # Calculate average loss for this epoch
        epoch_avg_loss = (epoch_train_loss + epoch_val_loss) / 2
        avg_losses.append(epoch_avg_loss)  # Store average loss for plotting

        # Print summary for this epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Training Loss: {epoch_train_loss:.4f}")
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        print(f"Average Loss: {epoch_avg_loss:.4f}\n")

        # Save the model with the best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss  # Update the best validation loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Save model weights
            model_save_path = os.path.join(model_base_path, f"{model_name}.pth")  # Define save path
            torch.save(best_model_wts, model_save_path)  # Save the model weights to file
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

    # Load the best model (with the lowest validation loss)
    model.load_state_dict(torch.load(os.path.join(model_base_path, f"{model_name}.pth")))

    # Test phase: Evaluate the model on the test set
    test_loss = 0.0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradients needed during testing
        test_bar = tqdm(test_loader, desc="Testing Best Model", leave=True)  # Progress bar
        for batch in test_bar:
            if use_phase_mag:
                # Unpack inputs, magnitude features, and targets if using phase magnitude
                inputs, magnitude_features, targets = batch
                magnitude_features = magnitude_features.to(device)
                model_input = (inputs.to(device), magnitude_features)
            else:
                # Otherwise, just unpack image and targets
                inputs, targets = batch
                model_input = (inputs.to(device),)

            targets = targets.to(device)  # Move targets to device

            # Forward pass for testing
            outputs = model(*model_input)
            loss = criterion(outputs, targets)  # Compute the test loss
            test_loss += loss.item() * inputs.size(0)  # Accumulate the loss for the batch
            test_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update progress bar with current loss

    # Calculate the final test loss
    final_test_loss = test_loss / len(test_loader.dataset)
    print(f"\nFinal Test Loss: {final_test_loss:.4f}")  # Print the final test loss

    # Plot the training history (training/validation loss over epochs)
    plot_training_history(train_losses, val_losses, final_test_loss, num_epochs, config)

    return model  # Return the trained model

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and move to device (GPU/CPU)
model = ResNetRegression(config=config, num_outputs=len(to_predict)).to(device)

# Initialize optimizer based on config
optimizer_name = config["training"]["optimizer"]
optimizer = getattr(optim, optimizer_name)(
    model.parameters(), lr=config["training"]["learning_rate"]
)

# Get number of epochs from config
num_epochs = config["training"]["num_epochs"]

# Initialize loss function
criterion = CustomRMSELoss()

# Print model architecture
print(model)

# Get number of epochs from config
num_epochs = config["training"]["num_epochs"]

# Print message indicating training start
print("\nTraining ResNet model...")

# Train and evaluate the model
# Calls the train_and_evaluate function with model, data loaders, optimizer, and loss function
model = train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, optimizer, criterion, device, config)