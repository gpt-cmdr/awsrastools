import os
import subprocess
import re
import shutil
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed



"""
Since this library is quite large, I have collected a table here with each function, its description, arguments, and returns.


AwsRasTools: A class containing static methods for working with HEC-RAS files and executing plans.

Session 2.1: Command Line Automation of HEC-RAS with Python

| Method | Description | Arguments | Returns |
|--------|-------------|-----------|---------|
| compute_hecras_plan | Execute a single HEC-RAS plan using subprocess. | hecras_exe_path (str), project_file (str), plan_file (str) | bool: True if successful, False otherwise |
| find_hecras_project_file | Find the appropriate HEC-RAS project file (.prj) in the given folder. | folder_path (str or Path) | Path: The full path of the selected .prj file or None if no suitable file is found |
| get_plan_entries | Get all plan entries from a project file. | project_file (str) | pd.DataFrame: A DataFrame containing plan numbers for all plans in the project |
| get_flow_entries | Get all flow entries from a project file. | project_file (str) | pd.DataFrame: A DataFrame containing flow numbers for all flow files in the project |
| get_unsteady_entries | Get all unsteady flow entries from a project file. | project_file (str) | pd.DataFrame: A DataFrame containing unsteady numbers for all unsteady flow files in the project |
| get_geom_entries | Get all geometry entries from a project file. | project_file (str) | pd.DataFrame: A DataFrame containing geometry numbers for all geometry files in the project |
| copy_geometry_from_template | Copy geometry files from a template, find the next geometry number, and update the project file accordingly. | project_folder (str), project_file (str), template_geom (str) | str: New geometry number (e.g., 'g03') |
| apply_geometry_to_plan | Apply a geometry file to a plan file. | plan_file (str), geometry_number (str), project_file (str) | None |
| copy_unsteady_from_template | Copy unsteady flow files from a template, find the next unsteady number, and update the project file accordingly. | project_folder (str), project_file (str), template_unsteady (str) | str: New unsteady flow number (e.g., 'u03') |
| apply_unsteady_to_plan | Apply an unsteady flow file to a plan file. | plan_file (str), unsteady_number (str), project_file (str) | None |
| get_project_name | Extract the project name from the given project path. | project_path (Path) | str: The project name derived from the file name without extension |
| get_next_available_number | Determine the first available number for plan, unsteady, steady, or geometry files from 01 to 99. | existing_numbers (pandas.Series) | str: First available number as a two-digit string |
| copy_plan_from_template | Create a new plan file based on a template and update the project file. | project_folder (str), project_name (str), template_plan (str), new_plan_shortid (str, optional) | str: New plan number |
| run_plans_parallel | Run HEC-RAS plans in parallel using ThreadPoolExecutor. | ras_plan_entries (pd.DataFrame), hecras_exe_path (str), project_file (str), max_workers (int), cores_per_run (int) | dict: Dictionary with plan numbers as keys and execution success as values |
| set_num_cores | Update the maximum number of cores to use in the HEC-RAS plan file. | plan_file (str), num_cores (int) | None |
| update_geompre_flags | Update the simulation plan file to modify the `Run HTab` and `UNET Use Existing IB Tables` settings. | file_path (str), run_htab_value (int), use_ib_tables_value (int) | None |



Session 2.2: Modifying Unsteady Flow Hydrographs

| Method | Description | Arguments | Returns |
|--------|-------------|-----------|---------|
| extract_cross_section_attributes | Extract Cross Section Attributes from HEC-RAS HDF5 file and display as a pandas DataFrame. | hdf_path (str) | pandas.DataFrame: DataFrame containing Cross Section Attributes |
| check_values_exist | Check if specified river, reach, and station exist in the cross_section_attributes DataFrame. | df (pandas.DataFrame), river (str), reach (str), station (str) | tuple: (river_exists, reach_exists, station_exists) |
| extract_time_data_stamp | Extract time data stamp from HEC-RAS HDF5 file. | hdf_path (str) | pandas.DataFrame: DataFrame containing timestamps |
| extract_water_surface_and_flow | Extract water surface and flow data for a specific station from HEC-RAS HDF5 file. | hdf_path (str), station_target (float) | pandas.DataFrame: DataFrame containing timestamps, water surface, and flow data |
| plot_water_surface_and_flow | Plot water surface and flow data. | df_plot (pandas.DataFrame), station_name (str) | None |
| read_unsteady_file | Read the unsteady file and return its contents as a list of lines. | file_path (str) | list: List of lines from the unsteady file |
| identify_tables | Identify the start and end of each table in the unsteady file. | lines (list) | list: List of tuples containing table information (table_name, start_line, end_line) |
| parse_fixed_width_table | Parse a fixed-width table into a pandas DataFrame. | lines (list), start (int), end (int) | pandas.DataFrame: DataFrame containing the parsed table data |
| extract_tables | Extract all tables from the unsteady file and return them as a dictionary of DataFrames. | file_path (str) | dict: Dictionary of pandas DataFrames containing the extracted tables |
| scale_flow_hydrograph | Scale the Flow Hydrograph values by a linear scale factor. | tables (dict), scale_factor (float) | tuple: (Updated tables dictionary with scaled Flow Hydrograph, Original Flow Hydrograph values) |
| write_table_to_file | Write the updated table back to the file in fixed-width format. | file_path (str), table_name (str), df (pandas.DataFrame), start_line (int) | None |
| plot_original_and_scaled | Plot the original and scaled Flow Hydrograph values. | original_values (pandas.Series), scaled_values (pandas.Series), scale_factor (float) | None |

Session 2.3: Applying 2D Infiltration Overrides

| Method | Description | Arguments | Returns |
|--------|-------------|-----------|---------|
| explore_hdf | Recursively explore and print HDF5 file structure. | file_path (str), max_depth (int, optional) | None |
| modify_infiltration_rate | Modify the infiltration rate for a specific land cover type in an HDF file. | hdf_file_path (str), land_cover_type (str), new_rate (float) | None |
| run_model | Run a HEC-RAS model for a specific plan. | project_path (str), plan_name (str) | bool: True if the model run was successful, False otherwise |
| save_results | Save the results of a model run with a specific naming convention. | project_path (str), plan_name (str), land_cover_type (str), infiltration_rate (float) | None |
| plot_wsel_timeseries | Plot water surface elevation time series for a specific cell ID from multiple HDF files. | hdf_paths (dict), specific_cell_id (int) | None |
"""

# Functions from Session 2.1 Command Line Automation of HEC-RAS with Python

class AwsRasTools:
    @staticmethod
    def compute_hecras_plan(hecras_exe_path, project_file, plan_file):
        """
        Execute a single HEC-RAS plan using subprocess.
        
        Parameters:
        hecras_exe_path (str): Path to HEC-RAS executable
        project_file (str): Full path to HEC-RAS project file (.prj)
        plan_file (str): Full path to HEC-RAS plan file (.p*)
        
        Returns:
        bool: True if successful, False otherwise

        Example:
        hecras_exe_path = r"C:\Program Files (x86)\HEC\HEC-RAS\6.3\RAS.exe"
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        plan_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.p01"
        success = AwsRasTools.compute_hecras_plan(hecras_exe_path, project_file, plan_file)
        """
        cmd = f'\"{hecras_exe_path}\" -c \"{project_file}\" \"{plan_file}\"'
        print(f"Running command: {cmd}")

        try:
            subprocess.run(cmd, check=True, shell=True)
            print(f"HEC-RAS is closed, check to ensure results are present: {os.path.basename(plan_file)}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running plan: {os.path.basename(plan_file)}")
            print(f"Error message: {str(e)}")
            return False

    @staticmethod
    def find_hecras_project_file(folder_path):
        """
        Find the appropriate HEC-RAS project file (.prj) in the given folder.
        
        This method searches for .prj files and uses various heuristics to select
        the most appropriate one, including checking for corresponding .rasmap files
        and searching for 'Proj Title=' within the .prj files.
        
        Parameters:
        folder_path (str or Path): Path to the folder containing HEC-RAS files.
        
        Returns:
        Path: The full path of the selected .prj file or None if no suitable file is found.

        Example:
        project_folder = r"C:\HEC-RAS_Projects\Muncie"
        project_file = AwsRasTools.find_hecras_project_file(project_folder)
        if project_file:
            print(f"Found project file: {project_file}")
        else:
            print("No suitable project file found.")
        """
        print(f"Starting to search for project file in folder: {folder_path}")
        folder_path = Path(folder_path)
        
        print("Searching for .prj files...")
        prj_files = list(folder_path.glob("*.prj"))
        print(f"Found {len(prj_files)} .prj files")
        
        print("Searching for .rasmap files...\n\n Begin Search Logic:\n")
        rasmap_files = list(folder_path.glob("*.rasmap"))
        print(f"Found {len(rasmap_files)} .rasmap files")
        
        if len(prj_files) == 1:
            project_file = prj_files[0]
            print(f"Only one .prj file found. Selecting: {project_file}")
        elif len(prj_files) > 1:
            if len(rasmap_files) == 1:
                project_file = folder_path / (rasmap_files[0].stem + ".prj")
                if project_file.exists():
                    print(f"Multiple .prj files found, but only one .rasmap file. Selecting corresponding .prj file: {project_file}")
                else:
                    print(f"Corresponding .prj file for .rasmap not found: {project_file}")
                    return None
            else:
                print("Multiple .prj and .rasmap files found. Searching for 'Proj Title=' in .prj files.")
                for prj_file in prj_files:
                    with open(prj_file, 'r') as f:
                        content = f.read()
                        if "Proj Title=" in content:
                            project_file = prj_file
                            print(f"Found 'Proj Title=' in file: {project_file}")
                            break
                else:
                    print("No .prj file with 'Proj Title=' found.")
                    return None
        else:
            print("No .prj files found in the specified folder.")
            return None
        
        print(f"Selected project file: {project_file}")
        return project_file

    @staticmethod
    def get_plan_entries(project_file):
        """
        Parse HEC-RAS project file and create dataframe for plan entries.

        This method reads the project file and extracts information about all plan files
        referenced in the project.

        Parameters:
        project_file (str): Full path to the HEC-RAS project file (.prj)

        Returns:
        pd.DataFrame: A DataFrame containing plan numbers for all plans in the project.

        Example:
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        plan_entries = AwsRasTools.get_plan_entries(project_file)
        print(plan_entries)
        """
        plan_entries = []
        with open(project_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Plan File="):
                    plan_number = line.split('=')[1].strip()[1:]
                    plan_entries.append({'plan_number': plan_number})
        return pd.DataFrame(plan_entries)

    @staticmethod
    def get_flow_entries(project_file):
        """
        Parse HEC-RAS project file and create dataframe for flow entries.

        This method reads the project file and extracts information about all flow files
        referenced in the project.

        Parameters:
        project_file (str): Full path to the HEC-RAS project file (.prj)

        Returns:
        pd.DataFrame: A DataFrame containing flow numbers for all flow files in the project.

        Example:
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        flow_entries = AwsRasTools.get_flow_entries(project_file)
        print(flow_entries)
        """
        flow_entries = []
        with open(project_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Flow File="):
                    flow_number = line.split('=')[1].strip()[1:]
                    flow_entries.append({'flow_number': flow_number})
        return pd.DataFrame(flow_entries)

    @staticmethod
    def get_unsteady_entries(project_file):
        """
        Parse HEC-RAS project file and create dataframe for unsteady entries.

        This method reads the project file and extracts information about all unsteady flow files
        referenced in the project.

        Parameters:
        project_file (str): Full path to the HEC-RAS project file (.prj)

        Returns:
        pd.DataFrame: A DataFrame containing unsteady numbers for all unsteady flow files in the project.

        Example:
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        unsteady_entries = AwsRasTools.get_unsteady_entries(project_file)
        print(unsteady_entries)
        """
        unsteady_entries = []
        with open(project_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Unsteady File="):
                    unsteady_number = line.split('=')[1].strip()[1:]
                    unsteady_entries.append({'unsteady_number': unsteady_number})
        return pd.DataFrame(unsteady_entries)

    @staticmethod
    def get_geom_entries(project_file):
        """
        Parse HEC-RAS project file and create dataframe for geometry entries.

        This method reads the project file and extracts information about all geometry files
        referenced in the project.

        Parameters:
        project_file (str): Full path to the HEC-RAS project file (.prj)

        Returns:
        pd.DataFrame: A DataFrame containing geometry numbers for all geometry files in the project.

        Example:
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        geom_entries = AwsRasTools.get_geom_entries(project_file)
        print(geom_entries)
        """
        geom_entries = []
        with open(project_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Geom File="):
                    geom_number = line.split('=')[1].strip()[1:]
                    geom_entries.append({'geom_number': geom_number})
        return pd.DataFrame(geom_entries)

    @staticmethod
    def copy_geometry_from_template(project_folder, project_file, template_geom):
        """
        Copy geometry files from a template, find the next geometry number,
        and update the project file accordingly. 

        This method creates a new geometry file based on an existing template,
        assigns it the next available geometry number, and updates the project file
        to include the new geometry.

        Parameters:
        - project_folder (str): Path to HEC-RAS project folder.
        - project_file (str): Path to HEC-RAS project file (.prj).
        - template_geom (str): Geometry number to be used as a template (e.g., 'g01').

        Returns:
        - str: New geometry number (e.g., 'g03').

        Raises:
        - FileNotFoundError: If the specified template geometry file does not exist.

        Example:
        project_folder = r"C:\HEC-RAS_Projects\Muncie"
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        template_geom = "g01"
        new_geom = AwsRasTools.copy_geometry_from_template(project_folder, project_file, template_geom)
        print(f"Created new geometry file: {new_geom}")
        """
        project_name = Path(project_file).stem
        
        template_geom_filename = f"{project_name}.{template_geom}"
        template_geom_path = os.path.join(project_folder, template_geom_filename)

        if not os.path.isfile(template_geom_path):
            raise FileNotFoundError(f"Template geometry file '{template_geom_path}' does not exist.")

        template_hdf_path = f"{template_geom_path}.hdf"
        if not os.path.isfile(template_hdf_path):
            raise FileNotFoundError(f"Template geometry .hdf file '{template_hdf_path}' does not exist.")

        with open(project_file, 'r') as file:
            lines = file.readlines()

        geom_file_pattern = re.compile(r'^Geom File=g(\d+)', re.IGNORECASE)
        existing_numbers = []

        for line in lines:
            match = geom_file_pattern.match(line.strip())
            if match:
                existing_numbers.append(int(match.group(1)))

        if existing_numbers:
            existing_numbers.sort()
            next_number = 1
            for num in existing_numbers:
                if num == next_number:
                    next_number += 1
                else:
                    break
        else:
            next_number = 1

        next_geom_number = f"{next_number:02d}"
        new_geom_filename = f"{project_name}.g{next_geom_number}"
        new_geom_path = os.path.join(project_folder, new_geom_filename)

        shutil.copyfile(template_geom_path, new_geom_path)
        print(f"Copied '{template_geom_path}' to '{new_geom_path}'.")

        new_hdf_path = f"{new_geom_path}.hdf"
        shutil.copyfile(template_hdf_path, new_hdf_path)
        print(f"Copied '{template_hdf_path}' to '{new_hdf_path}'.")

        new_geom_line = f"Geom File=g{next_geom_number}\n"

        insertion_index = None
        for i, line in enumerate(lines):
            match = geom_file_pattern.match(line.strip())
            if match:
                current_number = int(match.group(1))
                if current_number < next_number:
                    continue
                else:
                    insertion_index = i
                    break

        if insertion_index is not None:
            lines.insert(insertion_index, new_geom_line)
        else:
            header_pattern = re.compile(r'^(Proj Title|Current Plan|Default Exp/Contr|English Units)', re.IGNORECASE)
            header_indices = [i for i, line in enumerate(lines) if header_pattern.match(line.strip())]
            if header_indices:
                last_header_index = header_indices[-1]
                lines.insert(last_header_index + 2, new_geom_line)
            else:
                lines.insert(0, new_geom_line)

        with open(project_file, 'w') as file:
            file.writelines(lines)

        print(f"Inserted 'Geom File=g{next_geom_number}' into project file '{project_file}'.")

        return f"g{next_geom_number}"

    @staticmethod
    def apply_geometry_to_plan(plan_file, geometry_number, project_file):
        """
        Apply a geometry file to a plan file.
        
        This method updates the specified plan file to use the given geometry file.
        It first validates that the geometry number exists in the project file.

        Parameters:
        plan_file (str): Full path to the HEC-RAS plan file (.pXX).
        geometry_number (str): Geometry number to apply (e.g., 'g01').
        project_file (str): Full path to the project file to validate geometry number.
        
        Returns:
        None

        Raises:
        ValueError: If the specified geometry number is not found in the project file.

        Example:
        plan_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.p01"
        geometry_number = "g02"
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        AwsRasTools.apply_geometry_to_plan(plan_file, geometry_number, project_file)
        """
        geom_entries = AwsRasTools.get_geom_entries(project_file)
        
        if geometry_number not in geom_entries['geom_number'].values:
            raise ValueError(f"Geometry number {geometry_number} not found in project file.")
        
        with open(plan_file, 'r') as f:
            lines = f.readlines()
        
        with open(plan_file, 'w') as f:
            for line in lines:
                if line.startswith("Geom File=g"):
                    f.write(f"Geom File=g{geometry_number}\n")
                    print(f"Updated Geom File in {plan_file} to g{geometry_number}")
                else:
                    f.write(line)

    @staticmethod
    def copy_unsteady_from_template(project_folder, project_file, template_unsteady):
        """
        Copy unsteady flow files from a template, find the next unsteady number,
        and update the project file accordingly. 

        This method creates a new unsteady flow file based on an existing template,
        assigns it the next available unsteady number, and updates the project file
        to include the new unsteady flow file.

        Parameters:
        - project_folder (str): Path to HEC-RAS project folder.
        - project_file (str): Path to HEC-RAS project file (.prj).
        - template_unsteady (str): Unsteady flow number to be used as a template (e.g., 'u01').

        Returns:
        - str: New unsteady flow number (e.g., 'u03').

        Raises:
        - FileNotFoundError: If the specified template unsteady flow file does not exist.

        Example:
        project_folder = r"C:\HEC-RAS_Projects\Muncie"
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        template_unsteady = "u01"
        new_unsteady = AwsRasTools.copy_unsteady_from_template(project_folder, project_file, template_unsteady)
        print(f"Created new unsteady flow file: {new_unsteady}")
        """
        project_name = Path(project_file).stem
        
        template_unsteady_filename = f"{project_name}.{template_unsteady}"
        template_unsteady_path = os.path.join(project_folder, template_unsteady_filename)

        if not os.path.isfile(template_unsteady_path):
            raise FileNotFoundError(f"Template unsteady flow file '{template_unsteady_path}' does not exist.")

        with open(project_file, 'r') as file:
            lines = file.readlines()

        unsteady_file_pattern = re.compile(r'^Unsteady File=u(\d+)', re.IGNORECASE)
        existing_numbers = []

        for line in lines:
            match = unsteady_file_pattern.match(line.strip())
            if match:
                existing_numbers.append(int(match.group(1)))

        if existing_numbers:
            next_number = max(existing_numbers) + 1
        else:
            next_number = 1

        next_unsteady_number = f"{next_number:02d}"
        new_unsteady_filename = f"{project_name}.u{next_unsteady_number}"
        new_unsteady_path = os.path.join(project_folder, new_unsteady_filename)

        shutil.copyfile(template_unsteady_path, new_unsteady_path)
        print(f"Copied '{template_unsteady_path}' to '{new_unsteady_path}'.")

        template_hdf_path = f"{template_unsteady_path}.hdf"
        new_hdf_path = f"{new_unsteady_path}.hdf"

        if os.path.isfile(template_hdf_path):
            shutil.copyfile(template_hdf_path, new_hdf_path)
            print(f"Copied '{template_hdf_path}' to '{new_hdf_path}'.")
        else:
            print(f"No corresponding '.hdf' file found for '{template_unsteady_filename}'. Skipping '.hdf' copy.")

        new_unsteady_line = f"Unsteady File=u{next_unsteady_number}\n"

        insertion_index = None
        for i, line in enumerate(lines):
            match = unsteady_file_pattern.match(line.strip())
            if match:
                current_number = int(match.group(1))
                if current_number > next_number:
                    insertion_index = i
                    break

        if insertion_index is not None:
            lines.insert(insertion_index, new_unsteady_line)
        else:
            last_unsteady_index = max([i for i, line in enumerate(lines) if unsteady_file_pattern.match(line.strip())], default=-1)
            if last_unsteady_index != -1:
                lines.insert(last_unsteady_index + 1, new_unsteady_line)
            else:
                lines.append(new_unsteady_line)

        with open(project_file, 'w') as file:
            file.writelines(lines)

        print(f"Inserted 'Unsteady File=u{next_unsteady_number}' into project file '{project_file}'.")

        return f"u{next_unsteady_number}"

    @staticmethod
    def apply_unsteady_to_plan(plan_file, unsteady_number, project_file):
        """
        Apply an unsteady flow file to a plan file.
        
        This method updates the specified plan file to use the given unsteady flow file.
        It first validates that the unsteady number exists in the project file.

        Parameters:
        plan_file (str): Full path to the HEC-RAS plan file (.pXX).
        unsteady_number (str): Unsteady flow number to apply (e.g., 'u01').
        project_file (str): Full path to the project file to validate unsteady number.
        
        Returns:
        None

        Raises:
        ValueError: If the specified unsteady number is not found in the project file.

        Example:
        plan_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.p01"
        unsteady_number = "u02"
        project_file = r"C:\HEC-RAS_Projects\Muncie\Muncie.prj"
        AwsRasTools.apply_unsteady_to_plan(plan_file, unsteady_number, project_file)
        """
        unsteady_entries = AwsRasTools.get_unsteady_entries(project_file)
        
        if unsteady_number not in unsteady_entries['unsteady_number'].values:
            raise ValueError(f"Unsteady number {unsteady_number} not found in project file.")
        
        with open(plan_file, 'r') as f:
            lines = f.readlines()
        
        with open(plan_file, 'w') as f:
            for line in lines:
                if line.startswith("Flow File=u"):
                    f.write(f"Flow File=u{unsteady_number}\n")
                    print(f"Updated Flow File in {plan_file} to u{unsteady_number}")
                else:
                    f.write(line)
                    
                    
                    
                    

                    
                    
                    
                    
                    
          
    @staticmethod
    def get_project_name(project_path):
        """
        Extract the project name from the given project path.

        Parameters:
        project_path (Path): Path object representing the project file path

        Returns:
        str: The project name derived from the file name without extension

        Example:
        project_path = Path(r"C:\HEC-RAS_Projects\Muncie\Muncie.prj")
        project_name = AwsRasTools.get_project_name(project_path)
        print(f"Project name: {project_name}")
        """
        return Path(project_path).stem


    # NOTE: REVISED IN SESSION 2.4 HOMEWORK TO HANDLE NUMBERS GREATER THAN 09
    @staticmethod
    def get_next_available_number(existing_numbers):
        """
        Determine the first available number for plan, unsteady, steady, or geometry files.
        
        Parameters:
        existing_numbers (pandas.Series): Series of existing numbers as strings

        Returns:
        str: First available number as a string
        """
        existing_set = set(int(num[1:]) for num in existing_numbers if num[1:].isdigit())
        
        next_num = 1
        while next_num in existing_set:
            next_num += 1
        
        return f"{next_num:02d}"

    @staticmethod
    def copy_plan_from_template(project_folder, project_name, template_plan, new_plan_shortid=None):
        """
        Create a new plan file based on a template and update the project file.
        
        Parameters:
        project_folder (str): Path to HEC-RAS project folder
        project_name (str): Name of the HEC-RAS project (without extension)
        template_plan (str): Plan file to use as template (e.g., 'p01')
        new_plan_shortid (str, optional): New short identifier for the plan file
        
        Returns:
        str: New plan number

        Example:
        project_folder = r"C:\HEC-RAS_Projects\Muncie"
        project_name = "Muncie"
        template_plan = "p01"
        new_plan = AwsRasTools.copy_plan_from_template(project_folder, project_name, template_plan)
        print(f"Created new plan: {new_plan}")
        """
        project_file = os.path.join(project_folder, f"{project_name}.prj")

        if not os.path.isfile(project_file):
            raise FileNotFoundError(f"Project file not found: {project_file}")

        # Read the entire project file content
        with open(project_file, 'r') as file:
            project_content = file.read()

        # Find all plan numbers in the project file
        plan_numbers = re.findall(r'Plan File=p(\d+)', project_content)
        existing_numbers = [int(num) for num in plan_numbers]

        print(f"Existing plan numbers: {existing_numbers}")  # Debug print

        next_number = max(existing_numbers) + 1 if existing_numbers else 1
        new_plan_num = f"{next_number:02d}"

        print(f"Next plan number: {new_plan_num}")  # Debug print

        template_plan_path = os.path.join(project_folder, f"{project_name}.{template_plan}")
        new_plan_path = os.path.join(project_folder, f"{project_name}.p{new_plan_num}")
        
        shutil.copy(template_plan_path, new_plan_path)
        print(f"Copied {template_plan_path} to {new_plan_path}")

        with open(new_plan_path, 'r') as f:
            plan_lines = f.readlines()

        shortid_pattern = re.compile(r'^Short Identifier=(.*)$', re.IGNORECASE)
        for i, line in enumerate(plan_lines):
            match = shortid_pattern.match(line.strip())
            if match:
                current_shortid = match.group(1)
                new_shortid = (new_plan_shortid or (current_shortid + "_copy"))[:24]
                plan_lines[i] = f"Short Identifier={new_shortid}\n"
                break

        with open(new_plan_path, 'w') as f:
            f.writelines(plan_lines)

        print(f"Updated short identifier in {new_plan_path}")

        with open(project_file, 'r') as f:
            lines = f.readlines()

        new_plan_line = f"Plan File=p{new_plan_num}\n"
        updated_content = project_content + new_plan_line

        plan_file_pattern = re.compile(r'^Plan File=p(\d+)', re.IGNORECASE)
        insertion_index = None
        for i, line in enumerate(lines):
            match = plan_file_pattern.match(line.strip())
            if match:
                current_number = int(match.group(1))
                if current_number > next_number:
                    insertion_index = i
                    break

        if insertion_index is not None:
            lines.insert(insertion_index, new_plan_line)
        else:
            last_plan_index = max([i for i, line in enumerate(lines) if plan_file_pattern.match(line.strip())], default=-1)
            if last_plan_index != -1:
                lines.insert(last_plan_index + 1, new_plan_line)
            else:
                lines.append(new_plan_line)

        with open(project_file, 'w') as f:
            f.writelines(lines)

        print(f"Updated {project_file} with new plan p{new_plan_num}")

        return f"p{new_plan_num}"



    @staticmethod
    def run_plans_parallel(ras_plan_entries, hecras_exe_path, project_file, max_workers, cores_per_run):
        """
        Run HEC-RAS plans in parallel using ThreadPoolExecutor.
        
        Parameters:
        ras_plan_entries (pd.DataFrame): DataFrame containing plan file information
        hecras_exe_path (str): Path to HEC-RAS executable
        project_file (str): Full path to the project file
        max_workers (int): Maximum number of parallel runs
        cores_per_run (int): Number of cores to use per run
        
        Returns:
        dict: Dictionary with plan numbers as keys and execution success as values
        
        Example:
        ras_plan_entries = pd.DataFrame({
            'plan_number': ['01', '02'],
            'file_name': ['BaldEagle.p01', 'BaldEagle.p02'],
            'full_path': ['C:\\AWS_Session_2\\Bald Eagle Creek\\BaldEagle.p01',
                          'C:\\AWS_Session_2\\Bald Eagle Creek\\BaldEagle.p02']
        })
        hecras_exe_path = r"C:\Program Files (x86)\HEC\HEC-RAS\6.3\RAS.exe"
        project_file = r"C:\AWS_Session_2\Bald Eagle Creek\BaldEagle.prj"
        max_workers = 2
        cores_per_run = 2
        
        results = AwsRasTools.run_plans_parallel(ras_plan_entries, hecras_exe_path, project_file, max_workers, cores_per_run)
        print(results)
        # Expected output: {'01': True, '02': True}
        """
        
        def run_single_plan(plan_row, test_folder_path):
            """
            Execute a single HEC-RAS plan using the specified number of cores.
            
            Parameters:
            plan_row (pd.Series): A row from the DataFrame containing plan details
            test_folder_path (Path): Path to the test folder where the plan will be executed
            
            Returns:
            tuple: Plan number and execution success status
            """
            plan_number = plan_row['plan_number']
            file_name = plan_row['file_name']
            full_path = plan_row['full_path']
            
            # Update the plan file to use the specified number of cores
            AwsRasTools.set_num_cores(full_path, cores_per_run)
            
            # Construct the new path for the plan file in the test folder
            new_full_path = test_folder_path / Path(full_path).name
            print(f"Executing: Plan {plan_number}, File: {file_name}, Path: {new_full_path}")
            
            # Construct the command to run the HEC-RAS executable with the plan file
            cmd = f'"{hecras_exe_path}" -c "{new_full_path}"'
            print(f"Running command: {cmd}")
            try:
                # Execute the command and capture the output
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print(f"Completed: Plan {plan_number}, File: {file_name}")
                return plan_number, True
            except subprocess.CalledProcessError as e:
                # Handle any errors that occur during execution
                print(f"Failed: Plan {plan_number}, File: {file_name}")
                print(f"Error: {e.output}")
                return plan_number, False

        # Create multiple copies of the project folder for parallel execution
        project_folder = Path(project_file).parent
        test_folders = []
        for i in range(1, max_workers + 1):
            # Define the path for each test folder
            test_folder_path = project_folder.parent / f"{project_folder.name} [Test {i}]"
            # Copy the project folder to the test folder
            shutil.copytree(project_folder, test_folder_path, dirs_exist_ok=True)
            test_folders.append(test_folder_path)
            print(f"Created test folder: {test_folder_path}")

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Assign each plan to a specific test folder based on its index
            future_to_plan = {executor.submit(run_single_plan, row[1], test_folders[i % max_workers]): row[1]['plan_number'] for i, row in enumerate(ras_plan_entries.iterrows())}
            for future in as_completed(future_to_plan):
                # Retrieve the result of each completed future
                plan_number, success = future.result()
                results[plan_number] = success

        # Pause 3 seconds to allow files to close
        import time
        time.sleep(3)

        # Move all test folders back to the original test folder
        final_test_folder = project_folder.parent / f"{project_folder.name} [Test]"
        if not final_test_folder.exists():
            final_test_folder.mkdir()
        for test_folder in test_folders:
            for item in test_folder.iterdir():
                dest_path = final_test_folder / item.name
                if dest_path.exists():
                    # Remove existing directories or files at the destination path
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()
                # Move the item to the final test folder
                shutil.move(str(item), final_test_folder)
            # Remove the test folder after moving its contents
            shutil.rmtree(test_folder)
            print(f"Moved and removed test folder: {test_folder}")

        return results    
    
                    
    @staticmethod
    def set_num_cores(plan_file, num_cores):
        """
        Update the maximum number of cores to use in the HEC-RAS plan file.
        
        Parameters:
        plan_file (str): Full path to the plan file
        num_cores (int): Maximum number of cores to use
        
        Returns:
        None
        """
        d1_cores_pattern = re.compile(r"(UNET D1 Cores= )\d+")
        d2_cores_pattern = re.compile(r"(UNET D2 Cores= )\d+")
        ps_cores_pattern = re.compile(r"(PS Cores= )\d+")
        
        with open(plan_file, 'r') as file:
            content = file.read()
        
        # First update D2 cores
        new_content = d2_cores_pattern.sub(rf"\g<1>{num_cores}", content)
        
        # Check if D2 cores is 2, if so set others to 1, otherwise use num_cores
        d2_match = d2_cores_pattern.search(new_content)
        if d2_match and int(d2_match.group().split()[-1]) == 2:
            new_content = d1_cores_pattern.sub(r"\g<1>1", new_content)
            new_content = ps_cores_pattern.sub(r"\g<1>1", new_content)
            print(f"Updated {plan_file} with 2 cores for D2 and 1 core for D1 and PS.")
        else:
            new_content = d1_cores_pattern.sub(rf"\g<1>{num_cores}", new_content)
            new_content = ps_cores_pattern.sub(rf"\g<1>{num_cores}", new_content)
            print(f"Updated {plan_file} with {num_cores} cores for D1, D2, and PS.")
        
        with open(plan_file, 'w') as file:
            file.write(new_content)
        
        
        
    @staticmethod
    def update_geompre_flags(file_path, run_htab_value, use_ib_tables_value):
        """
        Update the simulation plan file to modify the `Run HTab` and `UNET Use Existing IB Tables` settings.

        Parameters:
        file_path (str): Path to the simulation plan file (.p06 or similar) that you want to modify.
        run_htab_value (int): Value for the `Run HTab` setting (0 or -1).
        use_ib_tables_value (int): Value for the `UNET Use Existing IB Tables` setting (0 or -1).

        Returns:
        None
        """
        if run_htab_value not in [-1, 0]:
            raise ValueError("Invalid value for `Run HTab`. Expected `0` or `-1`.")
        if use_ib_tables_value not in [-1, 0]:
            raise ValueError("Invalid value for `UNET Use Existing IB Tables`. Expected `0` or `-1`.")
        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            if line.strip().startswith("Run HTab="):
                updated_line = f"Run HTab= {run_htab_value} \n"
                updated_lines.append(updated_line)
            elif line.strip().startswith("UNET Use Existing IB Tables="):
                updated_line = f"UNET Use Existing IB Tables= {use_ib_tables_value} \n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)
    
    
    
    
                
# ---------------                 Functions from Session 2.2 Modifying Unsteady Flow Hydrographs              ---------------------------#


    @staticmethod
    def extract_cross_section_attributes(hdf_path):
        """
        Extract Cross Section Attributes from HEC-RAS HDF5 file and display as a pandas DataFrame.
        
        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file
        
        Returns:
        pandas.DataFrame: DataFrame containing Cross Section Attributes

        Example:
        hdf_path = r"Muncie_24Oct2024\Muncie.p01.hdf"
        cross_section_attributes = AwsRasTools.extract_cross_section_attributes(hdf_path)
        print(cross_section_attributes)
        """
        with h5py.File(hdf_path, 'r') as f:
            dataset_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Cross Section Attributes'
            dataset = f[dataset_path]
            
            data = dataset[:]
            column_names = dataset.dtype.names
            
            df = pd.DataFrame(data, columns=column_names)
            
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.decode('utf-8')
            
            return df

    @staticmethod
    def check_values_exist(df, river, reach, station):
        """
        Check if specified river, reach, and station exist in the cross_section_attributes DataFrame.

        Parameters:
        df (pandas.DataFrame): DataFrame containing cross-section attributes
        river (str): River name to check
        reach (str): Reach name to check
        station (str): Station value to check

        Returns:
        tuple: (river_exists, reach_exists, station_exists)

        Example:
        cross_section_attributes = AwsRasTools.extract_cross_section_attributes(hdf_path)
        river, reach, station = "White", "Muncie", "237.6455"
        river_exists, reach_exists, station_exists = AwsRasTools.check_values_exist(cross_section_attributes, river, reach, station)
        print(f"River '{river}' exists: {river_exists}")
        print(f"Reach '{reach}' exists: {reach_exists}")
        print(f"Station '{station}' exists: {station_exists}")
        """
        river_exists = river in df['River'].values
        reach_exists = reach in df['Reach'].values
        station_exists = station in df['Station'].astype(str).values
        
        return river_exists, reach_exists, station_exists

    @staticmethod
    def extract_time_data_stamp(hdf_path):
        """
        Extract time data stamp from HEC-RAS HDF5 file.

        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file

        Returns:
        pandas.DataFrame: DataFrame containing timestamps

        Example:
        hdf_path = r"Muncie_24Oct2024\Muncie.p01.hdf"
        df_timestamps = AwsRasTools.extract_time_data_stamp(hdf_path)
        print(df_timestamps.head())
        """
        from datetime import datetime

        def parse_ras_datetime(date_string):
            return datetime.strptime(date_string, "%d%b%Y %H:%M:%S")

        with h5py.File(hdf_path, 'r') as hdf:
            time_data_stamp = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][:]

        timestamps = [parse_ras_datetime(ts.decode('utf-8').strip()) for ts in time_data_stamp]
        return pd.DataFrame({'Timestamp': timestamps})

    @staticmethod
    def extract_water_surface_and_flow(hdf_path, station_target):
        """
        Extract water surface and flow data for a specific station from HEC-RAS HDF5 file.

        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file
        station_target (float): Target station value

        Returns:
        pandas.DataFrame: DataFrame containing timestamps, water surface, and flow data

        Example:
        hdf_path = r"Muncie_24Oct2024\Muncie.p01.hdf"
        station_target = 237.6455
        df_plot = AwsRasTools.extract_water_surface_and_flow(hdf_path, station_target)
        print(df_plot.head())
        """
        with h5py.File(hdf_path, 'r') as hdf:
            water_surface = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Water Surface'][:]
            flow = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Flow'][:]
            cross_section_attrs = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Cross Section Attributes'][:]

        cross_section_index = None
        for index, attr in enumerate(cross_section_attrs):
            station_value = float(attr['Station'].decode('utf-8').strip())
            if station_value == station_target:
                cross_section_index = index
                break

        if cross_section_index is None:
            raise ValueError(f"Station {station_target} not found in cross section attributes.")

        df_timestamps = AwsRasTools.extract_time_data_stamp(hdf_path)
        
        df_plot = pd.DataFrame({
            'Timestamp': df_timestamps['Timestamp'],
            'Water Surface': water_surface[:, cross_section_index],
            'Flow': flow[:, cross_section_index]
        })

        return df_plot

    @staticmethod
    def plot_water_surface_and_flow(df_plot, station_name):
        """
        Plot water surface and flow data.

        Parameters:
        df_plot (pandas.DataFrame): DataFrame containing timestamps, water surface, and flow data
        station_name (str): Name of the station for plot titles

        Returns:
        None

        Example:
        df_plot = AwsRasTools.extract_water_surface_and_flow(hdf_path, station_target)
        AwsRasTools.plot_water_surface_and_flow(df_plot, f"Station {station_target}")
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['Timestamp'], df_plot['Water Surface'], color='blue', marker='o', linestyle='-')
        plt.title(f'Water Surface at {station_name}')
        plt.xlabel('Time')
        plt.ylabel('Water Surface (ft)')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['Timestamp'], df_plot['Flow'], color='orange', marker='o', linestyle='-')
        plt.title(f'Flow at {station_name}')
        plt.xlabel('Time')
        plt.ylabel('Flow (cfs)')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def read_unsteady_file(file_path):
        """
        Read the unsteady file and return its contents as a list of lines.

        Parameters:
        file_path (str): Path to the unsteady file

        Returns:
        list: List of lines from the unsteady file

        Example:
        unsteady_file_path = r"Muncie_24Oct2024\Muncie.u01"
        lines = AwsRasTools.read_unsteady_file(unsteady_file_path)
        """
        with open(file_path, 'r') as file:
            return file.readlines()

    @staticmethod
    def identify_tables(lines):
        """
        Identify the start and end of each table in the unsteady file.

        Parameters:
        lines (list): List of lines from the unsteady file

        Returns:
        list: List of tuples containing table information (table_name, start_line, end_line)

        Example:
        lines = AwsRasTools.read_unsteady_file(unsteady_file_path)
        tables = AwsRasTools.identify_tables(lines)
        """
        table_types = ['Flow Hydrograph=', 'Gate Openings=', 'Stage Hydrograph=', 
                       'Uniform Lateral Inflow=', 'Lateral Inflow Hydrograph=']
        tables = []
        current_table = None
        for i, line in enumerate(lines):
            if any(table_type in line for table_type in table_types):
                if current_table:
                    tables.append((current_table[0], current_table[1], i-1))
                table_name = line.strip().split('=')[0] + '='
                num_values = int(line.strip().split('=')[1])
                current_table = (table_name, i+1, num_values)
        if current_table:
            tables.append((current_table[0], current_table[1], current_table[1] + (current_table[2] + 9) // 10))
        return tables

    @staticmethod
    def parse_fixed_width_table(lines, start, end):
        """
        Parse a fixed-width table into a pandas DataFrame.

        Parameters:
        lines (list): List of lines from the unsteady file
        start (int): Start line of the table
        end (int): End line of the table

        Returns:
        pandas.DataFrame: DataFrame containing the parsed table data

        Example:
        lines = AwsRasTools.read_unsteady_file(unsteady_file_path)
        tables = AwsRasTools.identify_tables(lines)
        df = AwsRasTools.parse_fixed_width_table(lines, tables[0][1], tables[0][2])
        """
        data = []
        for line in lines[start:end]:
            values = [line[i:i+8].strip() for i in range(0, len(line), 8)]
            parsed_values = []
            for value in values:
                try:
                    if len(value) > 8:
                        parsed_values.extend([float(value[:8]), float(value[8:])])
                    elif value:
                        parsed_values.append(float(value))
                except ValueError:
                    continue
            data.extend(parsed_values)
        return pd.DataFrame(data, columns=['Value'])

    @staticmethod
    def extract_tables(file_path):
        """
        Extract all tables from the unsteady file and return them as a dictionary of DataFrames.

        Parameters:
        file_path (str): Path to the unsteady file

        Returns:
        dict: Dictionary of pandas DataFrames containing the extracted tables

        Example:
        unsteady_file_path = r"Muncie_24Oct2024\Muncie.u01"
        tables = AwsRasTools.extract_tables(unsteady_file_path)
        for table_name, df in tables.items():
            print(f"\n{table_name}")
            print(df)
        """
        lines = AwsRasTools.read_unsteady_file(file_path)
        tables = AwsRasTools.identify_tables(lines)
        extracted_tables = {}
        for table_name, start, end in tables:
            df = AwsRasTools.parse_fixed_width_table(lines, start, end)
            extracted_tables[table_name] = df
        return extracted_tables

    @staticmethod
    def scale_flow_hydrograph(tables, scale_factor):
        """
        Scale the Flow Hydrograph values by a linear scale factor.

        Parameters:
        tables (dict): Dictionary containing extracted tables
        scale_factor (float): Factor to scale the Flow Hydrograph values

        Returns:
        tuple: (Updated tables dictionary with scaled Flow Hydrograph, Original Flow Hydrograph values)

        Example:
        tables = AwsRasTools.extract_tables(unsteady_file_path)
        scale_factor = 1.5
        tables, original_values = AwsRasTools.scale_flow_hydrograph(tables, scale_factor)
        """
        if "Flow Hydrograph=" not in tables:
            print("Flow Hydrograph table not found in the extracted tables.")
            return tables, None
        
        flow_df = tables["Flow Hydrograph="]
        original_values = flow_df['Value'].copy()
        flow_df['Value'] = (flow_df['Value'] * scale_factor).round().astype(int)
        tables["Flow Hydrograph="] = flow_df
        
        return tables, original_values

    @staticmethod
    def write_table_to_file(file_path, table_name, df, start_line):
        """
        Write the updated table back to the file in fixed-width format.

        Parameters:
        file_path (str): Path to the unsteady file
        table_name (str): Name of the table to update (e.g., "Flow Hydrograph=")
        df (pandas.DataFrame): DataFrame containing the updated values
        start_line (int): Line number where the table starts in the file

        Returns:
        None

        Example:
        AwsRasTools.write_table_to_file(unsteady_file_path, "Flow Hydrograph=", tables["Flow Hydrograph="], start_line)
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        formatted_values = []
        for i in range(0, len(df), 10):
            row = df['Value'].iloc[i:i+10]
            formatted_row = ''.join(f'{value:8d}' for value in row)
            formatted_values.append(formatted_row + '\n')
        
        lines[start_line:start_line+len(formatted_values)] = formatted_values
        
        with open(file_path, 'w') as file:
            file.writelines(lines)

    @staticmethod
    def plot_original_and_scaled(original_values, scaled_values, scale_factor):
        """
        Plot the original and scaled Flow Hydrograph values.

        Parameters:
        original_values (pandas.Series): Original Flow Hydrograph values
        scaled_values (pandas.Series): Scaled Flow Hydrograph values
        scale_factor (float): Factor used for scaling

        Returns:
        None

        Example:
        AwsRasTools.plot_original_and_scaled(original_values, scaled_values, scale_factor)
        """
        time_hours = np.arange(len(original_values))
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_hours, original_values, 'b-', label='Original')
        plt.plot(time_hours, scaled_values, 'r-', label='Scaled')
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Flow (cfs)')
        plt.title(f'Original vs Scaled Flow Hydrograph (Scale Factor: {scale_factor})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        max_value = scaled_values.max()
        max_index = np.argmax(scaled_values)
        plt.annotate(f'Max: {max_value}', xy=(max_index, max_value), 
                     xytext=(max_index, max_value + 10), 
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=10, color='red')
        
        plt.show()
        
        
# ------------------------------ Functions from Session 2.3 Applying 2D Infiltration Overrides ---------------------------#
               
            
    @staticmethod
    def explore_hdf(file_path, max_depth=None):
        """
        Recursively explore and print HDF5 file structure
        
        Parameters:
        file_path (str): Path to the HDF5 file
        max_depth (int, optional): Maximum depth to explore (None for no limit)
        
        Returns:
        None
        """
        def print_hdf_structure(name, obj):
            print(f"\nPath: {name}")
            print(f"Type: {type(obj).__name__}")
            
            if isinstance(obj, h5py.Dataset):
                print(f"Shape: {obj.shape}")
                print(f"Dtype: {obj.dtype}")
                print("Attributes:")
                for key, value in obj.attrs.items():
                    print(f"  {key}: {value}")

        def _recursive_explore(name, obj, current_depth=0):
            if max_depth is not None and current_depth > max_depth:
                return
            
            print_hdf_structure(name, obj)
            
            if isinstance(obj, h5py.Group):
                for key, value in obj.items():
                    _recursive_explore(f"{name}/{key}", value, current_depth + 1)

        try:
            with h5py.File(file_path, 'r') as hdf_file:
                hdf_file.visititems(lambda name, obj: _recursive_explore(name, obj))
        except Exception as e:
            print(f"Error exploring HDF file: {e}")

    @staticmethod
    def modify_infiltration_rate(hdf_file_path, land_cover_type, new_rate):
        """
        Modify the infiltration rate for a specific land cover type in an HDF file.

        Parameters:
        hdf_file_path (str): Path to the HDF file
        land_cover_type (str): Land cover type to modify
        new_rate (float): New infiltration rate value

        Returns:
        None
        """
        print(f"Starting modification of infiltration rate for {land_cover_type} to {new_rate}")
        with h5py.File(hdf_file_path, 'r+') as hdf_file:
            print(f"Opened HDF file: {hdf_file_path}")
            dataset = hdf_file['/Variables']
            print("Accessed '/Variables' dataset")
            
            original_chunks = dataset.chunks
            original_compression = dataset.compression
            original_compression_opts = dataset.compression_opts
            original_maxshape = dataset.maxshape
            print(f"Stored original dataset properties: chunks={original_chunks}, compression={original_compression}")
            
            names = [name.decode('utf-8') for name in dataset['Name'][:]]
            print(f"Converted dataset names to list of strings. Total names: {len(names)}")
            
            mask = np.array([land_cover_type in name for name in names])
            print(f"Created boolean mask. Matching entries: {np.sum(mask)}")
            
            new_data = dataset[:]
            
            new_data['Minimum Infiltration Rate'][mask] = new_rate
            print(f"Modified infiltration rate for {np.sum(mask)} entries")
            
            del hdf_file['/Variables']
            print("Deleted original '/Variables' dataset")
            
            hdf_file.create_dataset('/Variables', data=new_data, 
                                    chunks=original_chunks,
                                    compression=original_compression,
                                    compression_opts=original_compression_opts,
                                    maxshape=original_maxshape)
            print("Created new '/Variables' dataset with modified data")
            
        print("Verifying changes...")
        with h5py.File(hdf_file_path, 'r') as hdf_file:
            verify_dataset = hdf_file['/Variables']
            verify_names = [name.decode('utf-8') for name in verify_dataset['Name'][:]]
            verify_mask = np.array([land_cover_type in name for name in verify_names])
            
            modified_rates = verify_dataset['Minimum Infiltration Rate'][verify_mask]
            modified_names = np.array(verify_names)[verify_mask]
            
            print(f"Modified entries for {land_cover_type}:")
            for name, rate in zip(modified_names, modified_rates):
                print(f"{name}: {rate}")
            
            if np.all(modified_rates == new_rate):
                print(f"Verification successful. All matching entries have been updated to {new_rate}")
            else:
                print("Verification failed. Not all entries were updated correctly.")
        
        print(f"Successfully modified {land_cover_type} infiltration rate to {new_rate}")

    @staticmethod
    def run_model(project_path, plan_name):
        """
        Run a HEC-RAS model for a specific plan.

        Parameters:
        project_path (str): Path to the project folder
        plan_name (str): Name of the plan file (without extension)

        Returns:
        bool: True if the model run was successful, False otherwise
        """
        hecras_exe_path = r'C:\Program Files (x86)\HEC\HEC-RAS\6.6\RAS.exe'

        abs_project_path = os.path.abspath(project_path)
        project_file = os.path.join(abs_project_path, "BaldEagleDamBrk.prj")
        plan_file = os.path.join(abs_project_path, f"BaldEagleDamBrk.{plan_name}")
        success = AwsRasTools.compute_hecras_plan(hecras_exe_path, project_file, plan_file)
        if success:
            print(f"Completed model run for {plan_name}")
        else:
            print(f"Error running model for {plan_name}")
        return success

    @staticmethod
    def save_results(project_path, plan_name, land_cover_type, infiltration_rate):
        """
        Save the results of a model run with a specific naming convention.

        Parameters:
        project_path (str): Path to the project folder
        plan_name (str): Name of the plan file (without extension)
        land_cover_type (str): Type of land cover modified
        infiltration_rate (float): Infiltration rate used

        Returns:
        None
        """
        source = os.path.join(project_path, f"BaldEagleDamBrk.{plan_name}.hdf")
        destination = os.path.join(project_path, f"BaldEagleDamBrk_{land_cover_type}_{infiltration_rate:.2f}.{plan_name}.hdf")
        shutil.copy(source, destination)
        print(f"Saved results to {destination}")

    @staticmethod
    def plot_wsel_timeseries(hdf_paths, specific_cell_id=767):
        """
        Plots the water surface elevation time series for a specific cell ID from multiple HDF files with different infiltration rates.

        Parameters:
        hdf_paths (dict): Dictionary containing infiltration rates as keys and corresponding HDF file paths as values.
        specific_cell_id (int): The specific cell ID to plot the time series for. Default is 767.

        Returns:
        None
        """
        timeseries_data = {}

        for rate, path in hdf_paths.items():
            cells_timeseries_ds = HdfResultsMesh.mesh_cells_timeseries_output(path)
            timeseries_data[rate] = cells_timeseries_ds['BaldEagleCr']['Water Surface']

        plt.figure(figsize=(12, 6))

        for rate, water_surface in timeseries_data.items():
            time_values = water_surface.coords['time'].values
            wsel_timeseries = water_surface.sel(cell_id=specific_cell_id)
            peak_value = wsel_timeseries.max().item()
            peak_index = wsel_timeseries.argmax().item()

            plt.plot(time_values, wsel_timeseries, label=f'Cell ID: {specific_cell_id}, Rate: {rate}')
            plt.scatter(time_values[peak_index], peak_value, s=100, zorder=5, label=f'Peak at Rate: {rate}')
            plt.annotate(f'Peak: {peak_value:.2f} ft', 
                        (time_values[peak_index], peak_value),
                        xytext=(10, 10), textcoords='offset points',
                        ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.title(f'Water Surface Elevation Time Series for Specific Cell (ID: {specific_cell_id})')
        plt.xlabel('Time')
        plt.ylabel('Water Surface Elevation (ft)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        logging.info(f"Plotted water surface elevation time series for specific cell ID: {specific_cell_id} for all infiltration rates")

        plt.show()

        for rate, water_surface in timeseries_data.items():
            wsel_timeseries = water_surface.sel(cell_id=specific_cell_id)
            peak_value = wsel_timeseries.max().item()
            peak_index = wsel_timeseries.argmax().item()
            print(f"Statistics for Cell ID {specific_cell_id} at Infiltration Rate {rate}:")
            print(f"Minimum WSEL: {wsel_timeseries.min().item():.2f} ft")
            print(f"Maximum WSEL: {peak_value:.2f} ft")
            print(f"Mean WSEL: {wsel_timeseries.mean().item():.2f} ft")
            print(f"Time of peak: {time_values[peak_index]}")          
                      
            
    @staticmethod
    def extract_cross_section_attributes(hdf_path):
        """
        Extract Cross Section Attributes from HEC-RAS HDF5 file and display as a pandas DataFrame.
        
        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file
        
        Returns:
        pandas.DataFrame: DataFrame containing Cross Section Attributes
        """
        with h5py.File(hdf_path, 'r') as f:
            dataset_path = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Cross Section Attributes'
            dataset = f[dataset_path]
            
            data = dataset[:]
            column_names = dataset.dtype.names
            
            df = pd.DataFrame(data, columns=column_names)
            
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.decode('utf-8')
            
            return df

    @staticmethod
    def check_values_exist(df, river, reach, station):
        """
        Check if specified river, reach, and station exist in the cross_section_attributes DataFrame.

        Parameters:
        df (pandas.DataFrame): DataFrame containing cross-section attributes
        river (str): River name to check
        reach (str): Reach name to check
        station (str): Station value to check

        Returns:
        tuple: (river_exists, reach_exists, station_exists)
        """
        river_exists = river in df['River'].values
        reach_exists = reach in df['Reach'].values
        station_exists = station in df['Station'].astype(str).values
        
        return river_exists, reach_exists, station_exists

    @staticmethod
    def extract_time_data_stamp(hdf_path):
        """
        Extract time data stamp from HEC-RAS HDF5 file.

        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file

        Returns:
        pandas.DataFrame: DataFrame containing timestamps
        """
        from datetime import datetime

        def parse_ras_datetime(date_string):
            return datetime.strptime(date_string, "%d%b%Y %H:%M:%S")

        with h5py.File(hdf_path, 'r') as hdf:
            time_data_stamp = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][:]

        timestamps = [parse_ras_datetime(ts.decode('utf-8').strip()) for ts in time_data_stamp]
        return pd.DataFrame({'Timestamp': timestamps})

    @staticmethod
    def extract_water_surface_and_flow(hdf_path, station_target):
        """
        Extract water surface and flow data for a specific station from HEC-RAS HDF5 file.

        Parameters:
        hdf_path (str): Path to the HEC-RAS HDF5 file
        station_target (float): Target station value

        Returns:
        pandas.DataFrame: DataFrame containing timestamps, water surface, and flow data
        """
        with h5py.File(hdf_path, 'r') as hdf:
            water_surface = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Water Surface'][:]
            flow = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Flow'][:]
            cross_section_attrs = hdf['/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Cross Sections/Cross Section Attributes'][:]

        cross_section_index = None
        for index, attr in enumerate(cross_section_attrs):
            station_value = float(attr['Station'].decode('utf-8').strip())
            if station_value == station_target:
                cross_section_index = index
                break

        if cross_section_index is None:
            raise ValueError(f"Station {station_target} not found in cross section attributes.")

        df_timestamps = AwsRasTools.extract_time_data_stamp(hdf_path)
        
        df_plot = pd.DataFrame({
            'Timestamp': df_timestamps['Timestamp'],
            'Water Surface': water_surface[:, cross_section_index],
            'Flow': flow[:, cross_section_index]
        })

        return df_plot        
          
          
          
          
          
          
          
          
          
                    
""" 

Importing the AwsRasTools class and using the static methods
from awsrastools.awsrastools import AwsRasTools


Example usage of the AwsRasTools class
# Run Plan 01 of Muncie
hecras_exe_path = r"C:\Program Files (x86)\HEC\HEC-RAS\6.6\Ras.exe"
project_file = r"Muncie_24Oct2024\Muncie.prj"
# Get full path of project file
project_file = os.path.abspath(project_file)
plan_file = r"Muncie_24Oct2024\Muncie.p01"
# Get full path of plan file
plan_file = os.path.abspath(plan_file)

AwsRasTools.compute_hecras_plan(hecras_exe_path, project_file, plan_file)

"""
