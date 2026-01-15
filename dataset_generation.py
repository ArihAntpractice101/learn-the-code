"""
Telescope Mirror Alignment Dataset Generator

This script generates a dataset of point spread functions (PSFs) based on various mirror alignment parameters
for a multi-segment telescope. It uses Radiant Zemax 13 SP4 R2 through the PyZDDE package to simulate the optical
performance and stores the results in a MongoDB database.

The script configures the primary mirror segments (exactyly 6 of them) and the secondary mirror with
alignment parameters (piston, tip and tilt) and calculates the resulting PSF for each configuration.
It uses Sobol sequences for efficient parameter space sampling.

Requirements:

* PyMongo
* NumPy
* PyZDDE
* SciPy
* OmegaConf
* Zemax OpticStudio (must be running)

Configuration is loaded from a YAML file that specifies:

* MongoDB connection settings
* Zemax lens file and settings
* Mirror segment parameters (static, range or untrained\_range)
* Dataset size and starting index
"""

from pymongo import MongoClient
import numpy as np
import os
import pyzdde.zdde as pyz
from scipy.stats.qmc import Sobol
import warnings
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig
warnings.filterwarnings(action='ignore')

# Path to the configuration file that defines mirror parameters and dataset settings

config_path = './dataset_config.yaml'

# Check if configuration file exists

if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

# Load configuration from YAML file

config = OmegaConf.load(config_path)

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

# Connect to MongoDB

client = MongoClient(config['mongodb']['uri'])
db = client[config['mongodb']['database']]
full_collection = db[config['mongodb']['collection']]

# Create an index on the "index" field to ensure uniqueness

full_collection.create_index("index", name="idx", unique=True)

# Attempt to connect to Zemax using PyZDDE
try:
    ln = pyz.createLink()
except Exception:
    raise RuntimeError("Failed to create Zemax link. Check if Zemax is running and the DDE server is active.")

# Ensure the configuration includes the Zemax section
if 'zemax' not in config:
    raise KeyError("Zemax configuration section is missing in the configuration file.")

# Validate required keys in the Zemax configuration
if 'lens_file_path' not in config['zemax']:
    raise KeyError("Missing Zemax lens_file_path in the configuration.")
if not isinstance(config['zemax']['lens_file_path'], str):
    raise TypeError("Zemax lens_file_path must be a string.")

if 'config_file_path' not in config['zemax']:
    raise KeyError("Missing Zemax config_file_path in the configuration.")
if not isinstance(config['zemax']['config_file_path'], str):
    raise TypeError("Zemax config_file_path must be a string.")

if 'sampling' not in config['zemax']:
    raise KeyError("Missing Zemax sampling in the configuration.")
if not isinstance(config['zemax']['sampling'], int):
    raise TypeError("Zemax sampling must be an integer.")
if not (1 <= config['zemax']['sampling'] <= 10):
    raise ValueError("Zemax sampling must be an integer between 1 and 10.")

# Build the full path to the lens file using Zemax installation directory
try:
    zfile = os.path.join(
        ln.zGetPath()[1],  # Get the Zemax object path
        'Sequential',
        'Objectives',
        config['zemax']['lens_file_path']
    )

    # Check if the file exists before attempting to load
    if not os.path.exists(zfile):
        raise FileNotFoundError(f"Zemax lens file not found at the specified path: {zfile}")

    # Load the lens file into Zemax
    ln.zLoadFile(zfile)
except FileNotFoundError as fnf_err:
    raise fnf_err
except Exception:
    raise RuntimeError("Failed to load Zemax lens file.")

# Apply FFT PSF configuration using provided parameters
try:
    ln.zModifyFFTPSFSettings(
        settingsFile=config['zemax']['config_file_path'],
        dtype=None,
        sample=config['zemax']['sampling'],
        wave=None,
        field=None,
        surf=None,
        pol=None,
        norm=None,
        imgDelta=None
    )
except Exception:
    raise RuntimeError("Failed to modify Zemax FFT PSF settings.")

# Validate wavelength configuration

if 'wavelength' not in config:
    raise KeyError("Wavelength configuration is missing in the config file.")
if not isinstance(config['wavelength'], (int, float)):
    raise TypeError("Wavelength must be a number.")

# Set the wavelength for calculations

wavelength = config['wavelength']

def apply_mirror_alignment(
    primary_segment_1_piston, primary_segment_1_tip, primary_segment_1_tilt,
    primary_segment_2_piston, primary_segment_2_tip, primary_segment_2_tilt,
    primary_segment_3_piston, primary_segment_3_tip, primary_segment_3_tilt,
    primary_segment_4_piston, primary_segment_4_tip, primary_segment_4_tilt,
    primary_segment_5_piston, primary_segment_5_tip, primary_segment_5_tilt,
    primary_segment_6_piston, primary_segment_6_tip, primary_segment_6_tilt,
    secondary_mirror_piston, secondary_mirror_tip, secondary_mirror_tilt,
):
    """
    Apply mirror alignment parameters to the Zemax model and calculate the resulting PSF.

    This function takes alignment parameters for all six primary mirror segments and the secondary mirror,
    applies them to the Zemax model, and returns the resulting point spread function (PSF).

    Parameters:
    -----------
    primary_segment_X_piston : float
        Piston misalignment for primary mirror segment X in wavelength units / 1000
    primary_segment_X_tip : float
        Tip misalignment for primary mirror segment X in wavelength units / 125180
    primary_segment_X_tilt : float
        Tilt misalignment for primary mirror segment X in wavelength units / 125180
    secondary_mirror_piston : float
        Piston misalignment for the secondary mirror in wavelength units / 1000
    secondary_mirror_tip : float
        Tip misalignment for the secondary mirror in wavelength units / 61000
    secondary_mirror_tilt : float
        Tilt misalignment for the secondary mirror in wavelength units / 61000
        
    Returns:
    --------
    np.ndarray
        2D array containing the point spread function values
    """
    # Group segment parameters for easier processing
    segment_values = [
        (primary_segment_1_piston, primary_segment_1_tip, primary_segment_1_tilt),
        (primary_segment_2_piston, primary_segment_2_tip, primary_segment_2_tilt),
        (primary_segment_3_piston, primary_segment_3_tip, primary_segment_3_tilt),
        (primary_segment_4_piston, primary_segment_4_tip, primary_segment_4_tilt),
        (primary_segment_5_piston, primary_segment_5_tip, primary_segment_5_tilt),
        (primary_segment_6_piston, primary_segment_6_tip, primary_segment_6_tilt),
    ]

    # Apply parameters to each primary mirror segment
    for seg_idx, (piston_lambda, tip_lambda, tilt_lambda) in enumerate(segment_values):
        # Convert from wavelength units to physical units
        piston_mm = (piston_lambda / 1000) * wavelength  # Convert to mm
        tip_deg = np.degrees(np.arctan((wavelength * tip_lambda) / 125180))  # Convert to degrees
        tilt_deg = np.degrees(np.arctan((wavelength * tilt_lambda) / 125180))  # Convert to degrees
        
        # Set position parameters in the Zemax model
        # Surface 4 contains the primary mirror segments, with each segment being 2 objects apart
        ln.zSetNSCPosition(surfNum=4, objNum=seg_idx * 2 + 1, code=3, data=float(piston_mm))  # Z position (piston)
        ln.zSetNSCPosition(surfNum=4, objNum=seg_idx * 2 + 1, code=4, data=float(tip_deg))    # X tilt (tip)
        ln.zSetNSCPosition(surfNum=4, objNum=seg_idx * 2 + 1, code=5, data=float(tilt_deg))   # Y tilt (tilt)

    # Convert secondary mirror parameters from wavelength units to physical units
    sec_piston_mm = (secondary_mirror_piston / 1000) * wavelength  # Convert to mm
    sec_tip_deg = np.degrees(np.arctan((wavelength * secondary_mirror_tip) / 61000))  # Convert to degrees
    sec_tilt_deg = np.degrees(np.arctan((wavelength * secondary_mirror_tilt) / 61000))  # Convert to degrees

    # Apply parameters to the secondary mirror (surface 6)
    ln.zSetThickness(surfNum=6, value=float(sec_piston_mm))  # Z position (piston)
    ln.zSetSurfaceParameter(surfNum=6, param=3, value=float(sec_tip_deg))  # X tilt (tip)
    ln.zSetSurfaceParameter(surfNum=6, param=4, value=float(sec_tilt_deg))  # Y tilt (tilt)

    # Calculate the PSF using Fast Fourier Transform method
    psf_list = ln.zGetPSF('fft')[1]
    return np.array(psf_list)

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
                if not isinstance(value, (int, float)):
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

def generate_psf_dataset():
    """
    Generate a dataset of PSFs based on mirror alignment parameters.
    Stores both nominal and (optionally) defocused PSFs in MongoDB.
    """
    # Validate dataset configuration
    if 'dataset' not in config:
        raise KeyError("Missing 'dataset' section in the configuration.")

    if 'num_samples' not in config['dataset']:
        raise KeyError("Missing 'num_samples' key in the 'dataset' configuration section.")
    num_samples = config['dataset']['num_samples']

    if 'index_starts_with' not in config['dataset']:
        raise KeyError("Missing 'index_starts_with' key in the 'dataset' configuration section.")
    index_starts_with = config['dataset']['index_starts_with']

    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("`num_samples` must be a positive integer (≥ 1).")
    if not isinstance(index_starts_with, int) or index_starts_with < 1:
        raise ValueError("`index_starts_with` must be a positive integer (≥ 1).")
    
    if not isinstance(config["dataset"]["sampling_method"], str) :
        raise ValueError("`sampling_method` must be a string.")
    if config["dataset"]["sampling_method"] != "sobol":
        raise ValueError("`sampling_method` is currently implemented only as sobol for current system.")
    
    if 'seed' not in config['dataset']:
        raise KeyError("Seed value is missing from the configuration under 'dataset' section.")
    if not isinstance(config['dataset']['seed'], int):
        raise TypeError("`seed` value must be an integer.")
    
    validate_mirrors(config)

    # Parameter extraction
    sobol_parameters = []  # Sampled params
    static_parameters = {}  # Fixed segment params
    secondary_static_parameters = {}  # Fixed secondary params
    global_defocus = config['global_defocus']

    for segment in config['primary_mirror']['segments']:
        segment_id = segment['id']
        static_parameters[segment_id] = {}
        for param in ['piston', 'tip', 'tilt']:
            param_info = segment[param]
            if 'range' in param_info or 'untrained_range' in param_info:
                tag = 'range' if 'range' in param_info else 'untrained_range'
                sobol_parameters.append((segment_id, param, param_info[tag]))
            elif 'static' in param_info:
                static_parameters[segment_id][param] = param_info['static']

    secondary = config['secondary_mirror']
    for param in ['piston', 'tip', 'tilt']:
        param_info = secondary[param]
        if 'range' in param_info or 'untrained_range' in param_info:
            tag = 'range' if 'range' in param_info else 'untrained_range'
            sobol_parameters.append(('secondary', param, param_info[tag]))
        elif 'static' in param_info:
            secondary_static_parameters[param] = param_info['static']

    secondary_static_parameters['global_defocus'] = global_defocus

    sobol = Sobol(d=len(sobol_parameters), scramble=True, seed=config["dataset"]["seed"])
    sobol.fast_forward(index_starts_with)
    sobol_samples = sobol.random(n=num_samples)

    for sample_idx, sobol_sample in enumerate(sobol_samples):
        current_index = index_starts_with + sample_idx

        # Initialize segment dict
        segments = {}
        for sid in range(1, 7):
            segments[sid] = {}
            for param in ['piston', 'tip', 'tilt']:
                if sid in static_parameters and param in static_parameters[sid]:
                    segments[sid][param] = static_parameters[sid][param]

        # Initialize secondary
        secondary_params = {}
        for param in ['piston', 'tip', 'tilt']:
            if param in secondary_static_parameters:
                secondary_params[param] = secondary_static_parameters[param]
        secondary_params['global_defocus'] = secondary_static_parameters['global_defocus']

        # Apply Sobol sampled values
        for sobol_idx, (comp_id, param_name, val_range) in enumerate(sobol_parameters):
            low, high = val_range
            sampled_value = low + (high - low) * sobol_sample[sobol_idx]
            if comp_id == 'secondary':
                secondary_params[param_name] = sampled_value
            else:
                if comp_id not in segments:
                    segments[comp_id] = {}
                segments[comp_id][param_name] = sampled_value

        # PSF: Nominal
        psf_nominal = apply_mirror_alignment(
            primary_segment_1_piston = segments[1]['piston'], primary_segment_1_tip = segments[1]['tip'], primary_segment_1_tilt = segments[1]['tilt'],
            primary_segment_2_piston = segments[2]['piston'], primary_segment_2_tip = segments[2]['tip'], primary_segment_2_tilt = segments[2]['tilt'],
            primary_segment_3_piston = segments[3]['piston'], primary_segment_3_tip = segments[3]['tip'], primary_segment_3_tilt = segments[3]['tilt'],
            primary_segment_4_piston = segments[4]['piston'], primary_segment_4_tip = segments[4]['tip'], primary_segment_4_tilt = segments[4]['tilt'],
            primary_segment_5_piston = segments[5]['piston'], primary_segment_5_tip = segments[5]['tip'], primary_segment_5_tilt = segments[5]['tilt'],
            primary_segment_6_piston = segments[6]['piston'], primary_segment_6_tip = segments[6]['tip'], primary_segment_6_tilt = segments[6]['tilt'],
            secondary_mirror_piston = secondary_params['piston'],
            secondary_mirror_tip = secondary_params['tip'],
            secondary_mirror_tilt = secondary_params['tilt']
        )

        # Create primary dictionary (regular dict instead of OrderedDict)
        primary_dict = {}
        for k in sorted(segments):
            seg = segments[k]
            primary_dict[str(k)] = {
                "piston": seg["piston"],
                "tip": seg["tip"],
                "tilt": seg["tilt"]
            }

        # Create secondary dictionary (regular dict instead of OrderedDict)
        secondary_dict = {
            "piston": secondary_params["piston"],
            "tip": secondary_params["tip"],
            "tilt": secondary_params["tilt"],
            "global_defocus": secondary_params["global_defocus"]
        }

        # Construct MongoDB doc
        document = {
            "index": current_index,
            "primary": primary_dict,
            "secondary": secondary_dict,
            "psf_nominal_array": psf_nominal.tolist()
        }

        # PSF: Defocused (optional)
        if secondary_params['global_defocus'] != 0:
            psf_defocused = apply_mirror_alignment(
                primary_segment_1_piston = segments[1]['piston'], primary_segment_1_tip = segments[1]['tip'], primary_segment_1_tilt = segments[1]['tilt'],
                primary_segment_2_piston = segments[2]['piston'], primary_segment_2_tip = segments[2]['tip'], primary_segment_2_tilt = segments[2]['tilt'],
                primary_segment_3_piston = segments[3]['piston'], primary_segment_3_tip = segments[3]['tip'], primary_segment_3_tilt = segments[3]['tilt'],
                primary_segment_4_piston = segments[4]['piston'], primary_segment_4_tip = segments[4]['tip'], primary_segment_4_tilt = segments[4]['tilt'],
                primary_segment_5_piston = segments[5]['piston'], primary_segment_5_tip = segments[5]['tip'], primary_segment_5_tilt = segments[5]['tilt'],
                primary_segment_6_piston = segments[6]['piston'], primary_segment_6_tip = segments[6]['tip'], primary_segment_6_tilt = segments[6]['tilt'],
                secondary_mirror_piston = secondary_params['piston'] + secondary_params['global_defocus'],
                secondary_mirror_tip = secondary_params['tip'],
                secondary_mirror_tilt = secondary_params['tilt']
            )
            document["psf_defocused_array"] = psf_defocused.tolist()

        # Insert
        try:
            full_collection.insert_one(document)
            print(f"\rInserted document {sample_idx + 1}/{num_samples}", end='', flush=True)
        except Exception as e:
            print(f"\nError inserting document {sample_idx + 1}: {e}")
            break

    if sample_idx + 1 == num_samples:
        print("\nDataset generation complete.")
    else:
        print("\nDataset generation was not completed successfully. Do check the database and collection for complete information.")

if __name__ == "__main__":
    generate_psf_dataset()