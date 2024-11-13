import h5py

def h5_to_dict(h5_obj):
    """
    Recursively convert an h5py File or Group object into a nested dictionary.
    """
    result = {}
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)  # Recurse into groups
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Load dataset as NumPy array or scalar
    return result