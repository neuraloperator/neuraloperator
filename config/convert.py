#!/usr/bin/env python3
"""
Converter script for transforming YAML configuration files with inheritance
(using YAML anchors and references) to zencfg Python configuration format.

Usage: python convert.py yamlfile1 yamlfile2 ... yamlfileN
"""
import yaml
import argparse
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Set, Optional, Tuple


def camel_case(snake_str: str) -> str:
    """Convert snake_case to CamelCase but preserve underscores."""
    # Split by underscores, capitalize first letter of each component, then rejoin with underscores
    components = snake_str.split('_')
    return '_'.join(x.capitalize() for x in components)


def extract_anchor_refs(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract all anchor references used in the YAML file.
    Returns a dict mapping config name to its parent anchor name.
    """
    anchor_refs = {}
    for key, value in data.items():
        if isinstance(value, dict) and '<<' in value:
            # Find the parent in the << reference
            parent_ref = value.get('<<')
            if isinstance(parent_ref, str) and parent_ref.startswith('*'):
                parent_name = parent_ref[1:]  # Remove the '*'
                anchor_refs[key] = parent_name
    return anchor_refs


def is_scientific_notation(value: str) -> bool:
    """Check if a string represents a scientific notation number."""
    return bool(re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', value))


def get_python_type(value: Any) -> str:
    """Infer Python type from YAML value."""
    if value is None:
        return "Optional[Any]"
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        # Check if the string represents a scientific notation float
        if value.lower() in ['none', 'null']:
            return "Optional[Any]"
        elif is_scientific_notation(value):
            return "float"  # It's a scientific notation number like 1e-5
        else:
            return "str"
    elif isinstance(value, list):
        if not value:
            return "List[Any]"
        else:
            # Infer the type of the list elements
            elem_types = set()
            for elem in value:
                elem_types.add(get_python_type(elem))
            
            if len(elem_types) == 1:
                return f"List[{elem_types.pop()}]"
            else:
                return f"List[Union[{', '.join(sorted(elem_types))}]]"
    elif isinstance(value, dict):
        return "Dict[str, Any]"
    else:
        return "Any"


def find_nested_classes(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Find all nested dictionaries that should become classes.
    Returns a dict mapping class names to their data.
    """
    nested_dict_data = {}
    
    def process_nested(prefix: str, config_data: Dict[str, Any]):
        for key, value in config_data.items():
            if isinstance(value, dict) and key != '<<' and not key.startswith('*'):
                class_name = camel_case(key)
                nested_dict_data[class_name] = value
                process_nested(f"{prefix}.{key}" if prefix else key, value)
    
    for config_name, config_data in data.items():
        if config_name.startswith('*'):  # Skip anchor definitions
            continue
        process_nested("", config_data)
    
    return nested_dict_data


def process_config_value(key: str, value: Any, used_types: Set[str], 
                       nested_classes: Dict[str, str] = None) -> Tuple[str, str]:
    """Process a configuration value and return its type and formatted value."""
    nested_classes = nested_classes or {}
    
    if isinstance(value, dict) and '<<' in value:
        # Handle inheritance - skip the << key
        value = {k: v for k, v in value.items() if k != '<<'}
    
    if value is None or (isinstance(value, str) and value.lower() in ['none', 'null']):
        return "Optional[Any]", "None"
    elif isinstance(value, bool):
        return "bool", str(value)
    elif isinstance(value, (int, float)):
        return get_python_type(value), str(value)
    elif isinstance(value, str):
        # Check if the string represents a scientific notation float
        if is_scientific_notation(value):
            return "float", value  # Don't quote scientific notation
        else:
            return "str", f'"{value}"'
    elif isinstance(value, list):
        # Handle lists
        list_values = []
        elem_types = set()
        for item in value:
            elem_type = get_python_type(item)
            elem_types.add(elem_type)
            if isinstance(item, str):
                # Check if string item is scientific notation
                if is_scientific_notation(item):
                    list_values.append(item)  # Don't quote scientific notation
                else:
                    list_values.append(f'"{item}"')
            else:
                list_values.append(str(item))
        
        if len(elem_types) == 1:
            list_type = f"List[{elem_types.pop()}]"
        else:
            types_str = ', '.join(sorted(elem_types))
            list_type = f"List[Union[{types_str}]]"
            if "Union" not in used_types:
                used_types.add("Union")
        
        if "List" not in used_types:
            used_types.add("List")
        
        return list_type, f"[{', '.join(list_values)}]"
    elif isinstance(value, dict):
        # Check if this key maps to a nested class
        class_name = nested_classes.get(key)
        if class_name:
            return class_name, f"{class_name}()"
        else:
            # If we don't have a class for this dict, treat it as a regular dict
            if "Dict" not in used_types:
                used_types.add("Dict")
            return "Dict[str, Any]", str(value)
    else:
        if "Any" not in used_types:
            used_types.add("Any")
        return "Any", str(value)


def generate_config_class(name: str, config_data: Dict[str, Any], 
                         parent_class: str = "ConfigBase", 
                         anchor_refs: Dict[str, str] = None,
                         all_configs: Dict[str, Any] = None,
                         nested_classes: Dict[str, str] = None) -> Tuple[str, Set[str]]:
    """Generate a Python class definition from YAML config data."""
    class_name = camel_case(name)
    used_types = set()
    
    # Check if this config inherits from another
    parent_anchor = anchor_refs.get(name) if anchor_refs else None
    if parent_anchor and all_configs and parent_anchor in all_configs:
        parent_class = camel_case(parent_anchor)
    
    # Check if this is likely a subcategory (inherits from another config)
    if '<<' in config_data and isinstance(config_data['<<'], str):
        ref = config_data['<<']
        if ref.startswith('*'):
            parent_name = ref[1:]
            parent_class = camel_case(parent_name)
    
    lines = [f"class {class_name}({parent_class}):"]
    
    # Filter out the inheritance operator
    filtered_data = {k: v for k, v in config_data.items() if k != '<<'}
    
    if not filtered_data:
        lines.append("    pass")
        return '\n'.join(lines), used_types
    
    # Add fields
    for key, value in filtered_data.items():
        if key.startswith('*'):  # Skip anchor definitions
            continue
            
        field_type, field_value = process_config_value(key, value, used_types, nested_classes)
        lines.append(f"    {key}: {field_type} = {field_value}")
    
    return '\n'.join(lines), used_types


def convert_yaml_to_zencfg(yaml_content: str) -> str:
    """Convert YAML configuration to zencfg Python code."""
    # Load YAML with loader that handles anchors and aliases
    data = yaml.load(yaml_content, Loader=yaml.FullLoader)
    
    # Extract anchor references
    anchor_refs = extract_anchor_refs(data)
    
    # Find all nested dictionaries that should become classes
    nested_dict_data = find_nested_classes(data)
    nested_classes = {}
    
    # Generate classes for nested dictionaries
    for class_name, dict_data in nested_dict_data.items():
        class_code, used_types = generate_config_class(class_name, dict_data, "ConfigBase")
        nested_classes[class_name] = (class_code, used_types)
    
    # Separate base configs (those without parent references) from subcategories
    base_configs = []
    sub_configs = []
    
    for name, config in data.items():
        if name.startswith('*'):  # Skip anchor definitions
            continue
        if '<<' in config or name in anchor_refs.values():
            # This is either explicitly inheriting or is referenced as a parent
            base_configs.append(name)
        else:
            sub_configs.append(name)
    
    # Sort to ensure parents are defined before children
    ordered_configs = base_configs + sub_configs
    
    # Generate imports
    imports = ["from zencfg import ConfigBase"]
    all_used_types = set()
    
    # Create a mapping of key names to class names for nested dicts
    nested_class_map = {}
    for config_name in data:
        if config_name.startswith('*'):  # Skip anchor definitions
            continue
        for key, value in data[config_name].items():
            if isinstance(value, dict) and key != '<<' and not key.startswith('*'):
                nested_class_map[key] = camel_case(key)
    
    # Generate main classes
    main_classes = []
    
    for name in ordered_configs:
        if name.startswith('*'):
            continue
            
        class_code, types = generate_config_class(
            name, 
            data[name], 
            parent_class="ConfigBase", 
            anchor_refs=anchor_refs,
            all_configs=data,
            nested_classes=nested_class_map
        )
        
        all_used_types.update(types)
        main_classes.append(class_code)
    
    # Update all used types with types from nested classes
    for _, used_types in nested_classes.values():
        all_used_types.update(used_types)
    
    # Add nested classes before the main classes, starting with deepest nesting levels
    nested_class_codes = [code for code, _ in nested_classes.values()]
    classes = nested_class_codes + main_classes
    
    # Add type imports if needed
    typing_imports = []
    for t in sorted(all_used_types):
        if t in ["List", "Dict", "Union", "Optional", "Any"]:
            typing_imports.append(t)
    
    # Always include Optional and Any since we use them for None values
    if "Optional" not in typing_imports:
        typing_imports.append("Optional")
    if "Any" not in typing_imports:
        typing_imports.append("Any")
    
    if typing_imports:
        imports.insert(0, f"from typing import {', '.join(sorted(typing_imports))}")
    
    # Combine everything
    output_lines = imports + [""] + classes
    
    return '\n\n\n'.join(output_lines)


def main():
    parser = argparse.ArgumentParser(description='Convert YAML config files to zencfg format')
    parser.add_argument('yaml_files', nargs='+', help='YAML files to convert')
    
    args = parser.parse_args()
    
    for yaml_file in args.yaml_files:
        yaml_path = Path(yaml_file)
        
        if not yaml_path.exists():
            print(f"Error: File {yaml_file} not found")
            continue
        
        try:
            # Read YAML content
            with open(yaml_path, 'r') as f:
                yaml_content = f.read()
            
            # Convert to zencfg
            python_code = convert_yaml_to_zencfg(yaml_content)
            
            # Write output file
            output_path = yaml_path.with_suffix('.py')
            with open(output_path, 'w') as f:
                f.write(python_code)
            
            print(f"Converted {yaml_file} -> {output_path}")
            
        except Exception as e:
            print(f"Error converting {yaml_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()