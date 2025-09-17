import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Set, Optional, Tuple, Union
 


class AbaqusParsedFile(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__kwargs = {**kwargs}

    def __repr__(self, preview_str_len=100, preview_list_len=60, indent=4):
        def preview_str(s, preview=preview_str_len):
            s = '' if s is None else str(s)
            preview_slice = s[:preview]
            preview_text = preview_slice.replace("\n", "\\n")
            lines = s.count("\n") + (1 if s else 0)
            return f"{preview_text}... ({lines} lines, {len(s)} chars)"

        def preview_item(it, preview=preview_list_len):
            its = '' if it is None else str(it)
            preview_slice = its[:preview]
            preview_text = preview_slice.replace("\n", "\\n")
            lines = its.count("\n") + (1 if its else 0)
            return f"{preview_text}... ({lines} lines, {len(its)} chars)"

        out_lines = []
        pad = ' ' * indent

        for key, value in self.__kwargs.items():
            if isinstance(value, str):
                out_lines.append(f"{key}: {preview_str(value)}")
            elif isinstance(value, pd.DataFrame):
                out_lines.append(f"{key}: DataFrame({value.shape[0]} rows, {value.shape[1]} cols)")
                if not value.empty:
                    out_lines.append(f"{pad}columns: {list(value.columns)}")
                    for i, (idx, row) in enumerate(value.head(3).iterrows()):
                        if i < 2:
                            row_preview = str(row.to_dict())[:preview_list_len]
                            out_lines.append(f"{pad}- [{idx}] {row_preview}...")
            elif isinstance(value, dict):
                out_lines.append(f"{key}: dict with {len(value)} items")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        out_lines.append(f"{pad}- {sub_key}: DataFrame({sub_value.shape[0]} rows, {sub_value.shape[1]} cols)")
                    else:
                        out_lines.append(f"{pad}- {sub_key}: {type(sub_value).__name__}")
            elif isinstance(value, list):
                out_lines.append(f"{key}: [{len(value)} items]")
                if value and isinstance(value[0], list):
                    for sidx, item_list in enumerate(value[:3]):
                        out_lines.append(f"{pad}- [{sidx}]: [{len(item_list)} items]")
                        for lidx, item in enumerate(item_list[:3]):
                            out_lines.append(f"{pad*2}- [{lidx}] {preview_item(item)}")
                        if len(item_list) > 3:
                            out_lines.append(f"{pad*2}... and {len(item_list)-3} more items")
                    if len(value) > 3:
                        out_lines.append(f"{pad}... and {len(value)-3} more groups")
                else:
                    for idx, it in enumerate(value[:5]):
                        out_lines.append(f"{pad}- [{idx}] {preview_item(it)}")
                    if len(value) > 5:
                        out_lines.append(f"{pad}... and {len(value)-5} more items")
            else:
                sval = '' if value is None else str(value)
                preview_text = sval[:preview_str_len].replace("\n", "\\n")
                lines_count = sval.count("\n") + (1 if sval else 0)
                out_lines.append(f"{key}: {preview_text}... ({lines_count} lines, {len(sval)} chars)")

        return '\n'.join(out_lines)

class AbaqusInputFile:
    DELIMITER = '*'
    COMMENT = '**'

    def __init__(self, filename, workdir=None):
        self.filename = filename
        self.workdir = workdir if workdir else '.'
        self.__content = None
        self.__cleaned_content = None
        self.__commented_lines = None
        self.__splitted_content = None
        self.__tabular_data = None
        self.__nsets = None
        self.__elsets = None
        self.__surfaces = None
        self.__interp_data = None

    @property
    def content(self):
        if self.__content is None:
            self.__content = self.read_file()
        return self.__content
    
    @property
    def cleaned_content(self):
        if self.__cleaned_content is None:
            self.__cleaned_content = self.remove_comments_and_blank_lines(self.content)
        return self.__cleaned_content
    
    @property
    def commented_lines(self):
        if self.__commented_lines is None:
            _commented_lines = re.findall(r'\n\s*' + re.escape(self.COMMENT) + r'.*', self.content)
            self.__commented_lines = '\n'.join(_commented_lines).strip()
        return self.__commented_lines

    @property
    def splitted_content(self):
        if self.__splitted_content is None:
            self.__splitted_content = self.__split()
        return self.__splitted_content
    
    @property
    def tabular_data(self):
        if self.__tabular_data is None:
            self.__tabular_data = self.splitted_content['tabular_data']
        return self.__tabular_data

    @property
    def interp_data(self):
        if self.__interp_data is None:
            self.__interp_data = self.__interpolate_functions()
        return self.__interp_data

    @property
    def nsets(self) -> Dict[str, Set[int]]:
        if self.__nsets is None:
            self.__nsets = self._parse_sets('nset')
        return self.__nsets

    @property
    def elsets(self) -> Dict[str, Set[int]]:
        if self.__elsets is None:
            self.__elsets = self._parse_sets('elset')
        return self.__elsets

    @property
    def surfaces(self) -> Dict[str, List[List[int]]]:
        if self.__surfaces is None:
            self.__surfaces = self._parse_surfaces()
        return self.__surfaces

    def read_file(self):
        with open(self.filename, 'r') as file:
            return file.read()

    def remove_comments_and_blank_lines(self, content):
        content_no_comments = re.sub(r'^\s*' + re.escape(self.COMMENT) + r'.*$', '', content, flags=re.MULTILINE)
        content_no_blanks = "\n".join([line for line in content_no_comments.split('\n') if line.strip()])
        return content_no_blanks.strip()
    
    def _parse_tabular_data(self, block: str, keyword: str, data_types: List[type]) -> pd.DataFrame:
        lines = block.strip().split('\n')
        if not lines:
            return pd.DataFrame()
        
        data_lines = lines[1:]
        
        parsed_data = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            
            values = [v.strip() for v in line.split(',')]
            
            converted_row = []
            num_types = len(data_types)
            for i, value in enumerate(values):
                dtype = data_types[i] if i < num_types else str
                try:
                    converted_row.append(dtype(value))
                except (ValueError, TypeError):
                    converted_row.append(value)
            
            parsed_data.append(converted_row)
        
        return pd.DataFrame(parsed_data)

    def _group_material_blocks(self, blocks: List[str]) -> List[List[str]]:
        material_groups = []
        current_group = []
        
        material_keywords = ('MATERIAL', 'DEPVAR', 'USER MATERIAL', 'DENSITY', 'ELASTIC')

        for block in blocks:
            block_upper_start = block.strip().upper().split(',')[0].strip()
            if block_upper_start == 'MATERIAL':
                if current_group:
                    material_groups.append(current_group)
                current_group = [block]
            elif current_group and block_upper_start in material_keywords:
                current_group.append(block)
            elif current_group:
                material_groups.append(current_group)
                current_group = []

        if current_group:
            material_groups.append(current_group)
        
        return material_groups

    def _parse_sets(self, set_type: str) -> Dict[str, Set[int]]:
        keyword = set_type.upper()
        raw_blocks = self.splitted_content[f'{set_type}s']
        
        parsed_sets = {}
        deferred_sets = {}

        for block in raw_blocks:
            lines = [line.strip() for line in block.strip().split('\n')]
            header = lines[0]
            
            match = re.search(fr'{keyword}=([\w-]+)', header, re.IGNORECASE)
            if not match:
                continue
            set_name = match.group(1).upper()

            if 'GENERATE' in header.upper():
                if len(lines) > 1:
                    data_line = lines[1]
                    try:
                        start, end, step = map(int, [v.strip() for v in data_line.split(',')])
                        parsed_sets[set_name] = set(range(start, end + 1, step))
                    except ValueError:
                        print(f"Warning: Could not parse GENERATE parameters for {set_name}: {data_line}")
                continue

            data_lines = lines[1:]
            all_values = [v.strip() for line in data_lines for v in line.split(',') if v.strip()]
            
            numbers = set()
            references = []
            is_deferred = False
            for val in all_values:
                try:
                    numbers.add(int(val))
                except ValueError:
                    references.append(val.upper())
                    is_deferred = True
            
            if not is_deferred:
                parsed_sets[set_name] = numbers
            else:
                deferred_sets[set_name] = {'numbers': numbers, 'references': references}

        unresolved_deferred = deferred_sets
        resolved_in_pass = True
        while unresolved_deferred and resolved_in_pass:
            resolved_in_pass = False
            still_unresolved = {}
            for name, data in unresolved_deferred.items():
                combined_set = data['numbers'].copy()
                can_resolve_all_refs = True
                for ref_name in data['references']:
                    if ref_name in parsed_sets:
                        combined_set.update(parsed_sets[ref_name])
                    else:
                        can_resolve_all_refs = False
                        break
                
                if can_resolve_all_refs:
                    parsed_sets[name] = combined_set
                    resolved_in_pass = True
                else:
                    still_unresolved[name] = data
            unresolved_deferred = still_unresolved
            
        if unresolved_deferred:
            print(f"Warning: Could not resolve the following sets due to missing references: {list(unresolved_deferred.keys())}")

        return parsed_sets

    def _parse_surfaces(self) -> Dict[str, List[List[int]]]:
        raw_blocks = self.splitted_content.get('surfaces', [])
        if not raw_blocks:
            return {}

        elements_df = self.tabular_data.get('elements')
        if elements_df is None or elements_df.empty:
            print("Warning: Element data not found. Cannot parse surfaces.")
            return {}
        
        element_nodes_map = {row.element_id: row.node_ids for _, row in elements_df.iterrows()}
        
        face_maps = {
            # num_nodes: { side: [node_indices] }
            4: { # Tetrahedral (C3D4)
                'S1': [0, 1, 2], 'S2': [0, 3, 1], 'S3': [1, 3, 2], 'S4': [2, 3, 0]
            },
            6: { # Pentahedral/Wedge (C3D6)
                'S1': [0, 1, 2],    # Triangular face
                'S2': [3, 5, 4],    # Triangular face
                'S3': [0, 3, 4, 1], # Quad face
                'S4': [1, 4, 5, 2], # Quad face
                'S5': [2, 5, 3, 0]  # Quad face
            },
            8: { # Hexahedral/Brick (C3D8)
                'S1': [0, 1, 2, 3], 'S2': [4, 7, 6, 5], 'S3': [0, 4, 5, 1],
                'S4': [1, 5, 6, 2], 'S5': [2, 6, 7, 3], 'S6': [3, 7, 4, 0]
            }
        }

        parsed_surfaces = {}
        for block in raw_blocks:
            lines = [line.strip() for line in block.strip().split('\n')]
            header = lines[0]

            name_match = re.search(r'NAME=([\w-]+)', header, re.IGNORECASE)
            if not name_match: continue
            surface_name = name_match.group(1).upper()
            
            if 'TYPE=ELEMENT' not in header.upper(): continue

            surface_faces = []
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 2: continue
                
                elset_name, side = parts[0].upper(), parts[1].upper()

                element_ids = self.elsets.get(elset_name)
                if not element_ids:
                    print(f"Warning: Elset '{elset_name}' not found for surface '{surface_name}'.")
                    continue

                for elem_id in element_ids:
                    node_ids = element_nodes_map.get(elem_id)
                    if not node_ids:
                        print(f"Warning: Element ID {elem_id} not found in element data.")
                        continue
                    
                    num_nodes = len(node_ids)
                    if num_nodes in face_maps and side in face_maps[num_nodes]:
                        face_indices = face_maps[num_nodes][side]
                        face = [node_ids[i] for i in face_indices]
                        surface_faces.append(face)
                    else:
                        print(f"Warning: Unsupported element type ({num_nodes} nodes) or side ({side}) for element {elem_id}.")

            parsed_surfaces[surface_name] = surface_faces

        return parsed_surfaces

    def __interpolate_functions(self) -> Dict[str, Callable]:
        ampl = self.tabular_data.get('amplitude')
        if ampl is None or ampl.empty:
            print("Warning: Amplitude data not found. Cannot create interpolation function.")
            return {}
        try:
            from scipy.interpolate import interp1d
            time = ampl['time'].values
            amplitude = ampl['amplitude'].values
            interp_func = interp1d(time, amplitude, bounds_error=False, fill_value='extrapolate')
            return {'amplitude': interp_func}
        except ImportError:
            print("Warning: scipy.interpolate is not available. Cannot create interpolation function.")
            return {}
        

    def __split(self) -> AbaqusParsedFile:
        raw_blocks = self.cleaned_content.split(self.DELIMITER)
        blocks = [b.strip() for b in raw_blocks if b.strip()]
        
        categorized = {}
        
        keyword_map = {
            'STEP': 'steps',
            'END STEP': 'steps',
            'MATERIAL': 'materials',
            'DEPVAR': 'materials',
            'USER MATERIAL': 'materials',
            'DENSITY': 'materials',
            'ELASTIC': 'materials',
            'PLASTIC': 'materials',
            'DISTRIBUTION TABLE': 'distributions',
            'DISTRIBUTION': 'distributions',
            'AMPLITUDE': 'tabular_data',
            'NODE': 'tabular_data',
            'ELEMENT': 'tabular_data',
            'NSET': 'nsets',
            'ELSET': 'elsets',
            'SURFACE': 'surfaces',
        }
        
        for category in set(keyword_map.values()):
            categorized[category] = []
        categorized['full'] = [] 

        in_step = False
        for block in blocks:
            block_upper_start = block.strip().upper().split(',')[0].strip()
            
            if block_upper_start == 'STEP':
                in_step = True
                categorized['steps'].append([block])
                continue
            
            if in_step:
                categorized['steps'][-1].append(block)
                if block_upper_start == 'END STEP':
                    in_step = False
                continue

            category = keyword_map.get(block_upper_start)
            if category:
                categorized[category].append(block)
            else:
                categorized['full'].append(block)

        categorized['materials'] = self._group_material_blocks(categorized['materials'])
        
        tabular_blocks = categorized.pop('tabular_data', [])
        parsed_tabular_data = {}
        for block in tabular_blocks:
            block_upper = block.strip().upper()
            if block_upper.startswith('AMPLITUDE'):
                df = self._parse_tabular_data(block, 'AMPLITUDE', [float, float])
                if not df.empty: df.columns = ['time', 'amplitude']
                parsed_tabular_data['amplitude'] = df
            elif block_upper.startswith('NODE'):
                df = self._parse_tabular_data(block, 'NODE', [int, float, float, float])
                if not df.empty: df.columns = ['node_id', 'x', 'y', 'z']
                parsed_tabular_data['nodes'] = df
            elif block_upper.startswith('ELEMENT'):
                lines = block.strip().split('\n')
                if len(lines) > 1:
                    data_lines = lines[1:]
                    parsed_elements = []
                    for line in data_lines:
                        values = [v.strip() for v in line.strip().split(',') if v.strip()]
                        if values:
                            elem_id = int(values[0])
                            node_ids = [int(v) for v in values[1:]]
                            parsed_elements.append({'element_id': elem_id, 'node_ids': node_ids})
                    if parsed_elements:
                        parsed_tabular_data['elements'] = pd.DataFrame(parsed_elements)
        
        categorized['tabular_data'] = parsed_tabular_data

        return AbaqusParsedFile(**categorized)
    
    def get_step_boundaries(self, step_index: int = 0) -> Dict[str, Dict]:
        if not self.splitted_content.get('steps') or step_index >= len(self.splitted_content['steps']):
            return {}
        
        step_content = self.splitted_content['steps'][step_index]
        
        boundary_lines = []
        for block in step_content:
            if 'BOUNDARY' in block.upper():
                lines = block.split('\n')
                boundary_lines.extend(lines[1:])
        
        if not boundary_lines:
            return {}
        
        boundary_nodes_dict = {}
        for line in boundary_lines:
            line = line.strip()
            if not line:
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
                
            node_or_set = parts[0]
            param1 = int(parts[1])
            param2 = int(parts[2])
            
            if node_or_set.isdigit():
                nodes = [int(node_or_set)]
            else:
                # It's a node set name - get nodes from nsets
                nodes = list(self.nsets.get(node_or_set.upper(), set()))
            
            boundary_nodes_dict[node_or_set] = {
                'nodes': nodes,
                'dof_start': param1,
                'dof_end': param2
            }
        
        return boundary_nodes_dict
    
    def get_step_loads(self, step_index: int = 0) -> List[Dict]:
        if not self.splitted_content.get('steps') or step_index >= len(self.splitted_content['steps']):
            return []
        
        step_content = self.splitted_content['steps'][step_index]
        
        loads = []
        for block in step_content:
            if 'DSLOAD' in block.upper():
                lines = block.split('\n')
                for line in lines[1:]:  # Skip the DSLOAD keyword line
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        loads.append({
                            'surface': parts[0],
                            'type': parts[1],
                            'magnitude': float(parts[2])
                        })
        
        return loads


class DatFileParser:
    def __init__(self, fpath):
        self.__fpath = fpath
        self.__content = None
        self.__steps_data = None
        self.__discovered_node_sets = None

    @property
    def dat_file(self):
        return self.__fpath
    
    @property
    def content(self):
        if self.__content is None:
            self.__content = self.read_file()
        return self.__content

    def discover_node_sets(self) -> List[str]:
        if self.__discovered_node_sets is not None:
            return self.__discovered_node_sets
        
        content = self.content

        node_set_pattern = r'THE FOLLOWING TABLE IS PRINTED FOR NODES BELONGING TO NODE SET (\S+)'
        
        matches = re.findall(node_set_pattern, content)

        self.__discovered_node_sets = list(set(matches))
        
        return self.__discovered_node_sets
    
    def read_file(self):
        with open(self.dat_file, 'r') as file:
            return file.read()
    
    def parse_steps(self, node_sets: List[str] = None) -> Dict[int, List[Dict]]:
        if node_sets is None:
            node_sets = self.discover_node_sets()
            
        if self.__steps_data is None or node_sets != getattr(self, '_last_used_node_sets', []):
            self._last_used_node_sets = node_sets
            self.__steps_data = None
            
        if self.__steps_data is not None:
            return self.__steps_data
            
        content = self.content
        self.__steps_data = {}
        
        step_pattern = r'STEP\s+(\d+)\s+INCREMENT\s+\d+'
        step_matches = list(re.finditer(step_pattern, content))
        
        for i, match in enumerate(step_matches):
            step_num = int(match.group(1))
            step_start = match.start()
            
            if i + 1 < len(step_matches):
                step_end = step_matches[i + 1].start()
            else:
                step_end = len(content)
            
            step_content = content[step_start:step_end]
            
            increments = self._parse_increments_in_step(step_content, step_num, node_sets)
            self.__steps_data[step_num] = increments
            
        return self.__steps_data
    
    def _parse_increments_in_step(self, step_content: str, step_num: int, node_sets: List[str] = None) -> List[Dict]:
        if node_sets is None:
            node_sets = self.discover_node_sets()
        
        increments = []
        
        increment_pattern = r'INCREMENT\s+(\d+)\s+SUMMARY'
        increment_matches = list(re.finditer(increment_pattern, step_content))
        
        for i, match in enumerate(increment_matches):
            increment_num = int(match.group(1))
            increment_start = match.start()
            
            if i + 1 < len(increment_matches):
                increment_end = increment_matches[i + 1].start()
            else:
                increment_end = len(step_content)
            
            increment_content = step_content[increment_start:increment_end]
            
            increment_data = self._parse_single_increment(
                increment_content, step_num, increment_num, node_sets
            )
            
            if increment_data:
                increments.append(increment_data)
        
        return increments
    
    def _parse_single_increment(self, increment_content: str, step_num: int, increment_num: int, node_sets: List[str] = None) -> Optional[Dict]:
        if node_sets is None:
            node_sets = self.discover_node_sets()
        
        time_data = self._extract_time_data(increment_content)
        
        all_node_data = []
        for node_set_name in node_sets:
            node_data = self._extract_node_set_data(increment_content, node_set_name)
            if not node_data.empty:
                all_node_data.append(node_data)
        
        if not all_node_data:
            return None
        
        combined_node_data = pd.concat(all_node_data, ignore_index=True)
        
        return {
            'step': step_num,
            'increment': increment_num,
            'time_increment': time_data.get('time_increment', 0.0),
            'step_time': time_data.get('step_time', 0.0),
            'total_time': time_data.get('total_time', 0.0),
            'fraction_completed': time_data.get('fraction_completed', 0.0),
            'node_data': combined_node_data
        }
    
    def _extract_time_data(self, increment_content: str) -> Dict[str, float]:
        time_data = {}
        
        time_pattern = r'TIME INCREMENT COMPLETED\s+([\d\.\-E\+]+),\s+FRACTION OF STEP COMPLETED\s+([\d\.\-E\+]+)'
        time_match = re.search(time_pattern, increment_content)
        
        if time_match:
            time_data['time_increment'] = float(time_match.group(1))
            time_data['fraction_completed'] = float(time_match.group(2))
        
        step_time_pattern = r'STEP TIME COMPLETED\s+([\d\.\-E\+]+),\s+TOTAL TIME COMPLETED\s+([\d\.\-E\+]+)'
        step_time_match = re.search(step_time_pattern, increment_content)
        
        if step_time_match:
            time_data['step_time'] = float(step_time_match.group(1))
            time_data['total_time'] = float(step_time_match.group(2))
        
        return time_data
    
    def _extract_node_set_data(self, increment_content: str, node_set_name: str = "ENDO_NODES") -> pd.DataFrame:
        endo_pattern = fr'THE FOLLOWING TABLE IS PRINTED FOR NODES BELONGING TO NODE SET {re.escape(node_set_name)}.*?NODE FOOT-.*?U1\s+U2\s+U3.*?NOTE\s*(.*?)(?=\n\s*MAXIMUM|\n\s*\n|\Z)'
        
        match = re.search(endo_pattern, increment_content, re.DOTALL)
        
        if not match:
            return pd.DataFrame()
        
        node_section = match.group(1)
        
        nodes = []
        u1_values = []
        u2_values = []
        u3_values = []
        
        # Pattern for node data: node_number followed by three displacement values
        node_pattern = r'(\d+)\s+([\d\.\-E\+]+)\s+([\d\.\-E\+]+)\s+([\d\.\-E\+]+)'
        
        for line in node_section.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.search(node_pattern, line)
            if match:
                nodes.append(int(match.group(1)))
                u1_values.append(float(match.group(2)))
                u2_values.append(float(match.group(3)))
                u3_values.append(float(match.group(4)))
        
        if not nodes:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'NODE_SET': [node_set_name] * len(nodes),
            'NODE': nodes,
            'U1': u1_values,
            'U2': u2_values,
            'U3': u3_values
        })
    
    def get_step_data(self, step_num: int, node_sets: List[str] = None) -> List[Dict]:
        steps_data = self.parse_steps(node_sets)
        return steps_data.get(step_num, [])
    
    def get_increment_dataframe(self, step_num: int, increment_num: int, node_sets: List[str] = None) -> pd.DataFrame:
        step_data = self.get_step_data(step_num, node_sets)
        
        for increment_data in step_data:
            if increment_data['increment'] == increment_num:
                return increment_data['node_data']
        
        return pd.DataFrame()
    
    def get_all_increments_dataframe(self, step_num: int, node_sets: List[str] = None) -> pd.DataFrame:
        step_data = self.get_step_data(step_num, node_sets)
        
        if not step_data:
            return pd.DataFrame()
        
        all_data = []
        
        for increment_data in step_data:
            node_df = increment_data['node_data'].copy()
            
            if not node_df.empty:
                # Add time and increment information to each row
                node_df['step'] = increment_data['step']
                node_df['increment'] = increment_data['increment']
                node_df['total_time'] = increment_data['total_time']
                node_df['step_time'] = increment_data['step_time']
                node_df['time_increment'] = increment_data['time_increment']
                node_df['fraction_completed'] = increment_data['fraction_completed']
                
                all_data.append(node_df)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Reorder columns to put time info first
            time_cols = ['step', 'increment', 'total_time', 'step_time', 
                        'time_increment', 'fraction_completed']
            node_cols = ['NODE_SET', 'NODE', 'U1', 'U2', 'U3']
            return result[time_cols + node_cols]
        
        return pd.DataFrame()
    
    def get_all_steps_dataframe(self, node_sets: List[str] = None) -> pd.DataFrame:
        steps_data = self.parse_steps(node_sets)
        
        if not steps_data:
            return pd.DataFrame()
        
        all_step_data = []
        
        for step_num in sorted(steps_data.keys()):
            step_df = self.get_all_increments_dataframe(step_num, node_sets)
            if not step_df.empty:
                all_step_data.append(step_df)
        
        if all_step_data:
            return pd.concat(all_step_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def summary(self, node_sets: List[str] = None) -> Dict:
        if node_sets is None:
            node_sets = self.discover_node_sets()
            
        steps_data = self.parse_steps(node_sets)
        
        summary = {
            'total_steps': len(steps_data),
            'discovered_node_sets': self.discover_node_sets(),
            'used_node_sets': node_sets,
            'steps': {}
        }
        
        for step_num, increments in steps_data.items():
            summary['steps'][step_num] = {
                'total_increments': len(increments),
                'increments': [inc['increment'] for inc in increments]
            }
        
        return summary


class AbaqusVolumeAnalyzer:
    def __init__(self, inp_file: str, dat_file: str, workdir: Optional[str] = None):
        self.abaqus_input = AbaqusInputFile(inp_file, workdir)
        self.dat_parser = DatFileParser(dat_file)
        self._cached_volumes = {}

    def _step_index_from_num(self, step_num: Optional[int]) -> Optional[int]:
        if step_num is None:
            return None
        return max(0, int(step_num) - 1)

    def _get_amplitude_function(self) -> Optional[Callable[[float], float]]:
        try:
            return self.abaqus_input.interp_data.get('amplitude', None)
        except Exception:
            return None

    def _get_base_pressure_magnitude(self, step_num: Optional[int]) -> Optional[float]:
        try:
            idx = self._step_index_from_num(step_num)
            if idx is None:
                return None
            loads = self.abaqus_input.get_step_loads(idx) or []
            for ld in loads:
                if str(ld.get('type', '')).upper().startswith('P'):
                    return float(ld.get('magnitude'))
            return None
        except Exception:
            return None

    def _compute_effective_pressure(self, step_num: Optional[int], inc_meta: Optional[Dict]) -> Optional[float]:
        if inc_meta is None:
            return None

        p_fun = self._get_amplitude_function()
        base = self._get_base_pressure_magnitude(step_num)

        t_total = inc_meta.get('total_time', None)
        frac   = inc_meta.get('fraction_completed', None)

        amp_val = None
        if p_fun is not None and t_total is not None:
            try:
                amp_val = float(p_fun(float(t_total)))
            except Exception:
                amp_val = None

        if base is not None:
            if amp_val is not None:
                return base * amp_val
            if frac is not None:
                try:
                    return base * float(frac)
                except Exception:
                    return base
            return base

        return amp_val

    @staticmethod
    def volume_from_surface(points: np.ndarray, faces: List[List[int]], 
                            reference_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
        ref = np.array(reference_point, dtype=float)
        volume = 0.0

        for face in faces:
            if len(face) == 3:
                tri_faces = [face]
            elif len(face) == 4:
                tri_faces = [[face[0], face[1], face[2]], [face[0], face[2], face[3]]]
            else:
                continue

            for tri_face in tri_faces:
                tri = points[np.array(tri_face, dtype=int)]
                v1, v2, v3 = tri - ref
                tet_vol = float(np.dot(v1, np.cross(v2, v3))) / 6.0
                volume += tet_vol

        return abs(volume)

    def analyze_surface_volume(self, 
                               surface_name: str, 
                               reference_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                               step_num: Optional[int] = None, 
                               increment_num: Optional[int] = None) -> Dict:
        surface_name_upper = surface_name.upper()
        surfaces = self.abaqus_input.surfaces
        if surface_name_upper not in surfaces:
            available = list(surfaces.keys())
            raise ValueError(f"Surface '{surface_name}' not found. Available surfaces: {available}")

        faces = surfaces[surface_name_upper]

        nodes_df = self.abaqus_input.tabular_data.get("nodes", pd.DataFrame())
        if nodes_df.empty:
            raise ValueError("Nodes table is empty; cannot compute volume.")

        node_coords = nodes_df.set_index("node_id")[["x", "y", "z"]].to_dict("index")
        node_ids = sorted({nid for face in faces for nid in face})
        try:
            original_points = np.array([[node_coords[nid]["x"], node_coords[nid]["y"], node_coords[nid]["z"]]
                                        for nid in node_ids], dtype=float)
        except KeyError as e:
            missing = int(str(e).strip("'"))
            raise ValueError(f"Node {missing} referenced by surface '{surface_name_upper}' is missing from nodes table.")

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        local_faces = [[id_to_idx[nid] for nid in face] for face in faces]

        original_volume = self.volume_from_surface(original_points, local_faces, reference_point)

        displaced_volume = None
        volume_change = None

        disp_df: pd.DataFrame = pd.DataFrame()

        if step_num is not None and increment_num is not None:
            disp_df = self.dat_parser.get_increment_dataframe(step_num, increment_num)
            if not disp_df.empty:
                disp_dict = disp_df.set_index("NODE")[["U1", "U2", "U3"]].to_dict("index")

                displaced_points = []
                for nid in node_ids:
                    base = original_points[id_to_idx[nid]]
                    if nid in disp_dict:
                        u = np.array([disp_dict[nid]["U1"], disp_dict[nid]["U2"], disp_dict[nid]["U3"]], dtype=float)
                    else:
                        u = np.zeros(3, dtype=float)
                    displaced_points.append(base + u)

                displaced_points = np.array(displaced_points, dtype=float)
                displaced_volume = self.volume_from_surface(displaced_points, local_faces, reference_point)
                volume_change = displaced_volume - original_volume

        inc_meta = None
        if step_num is not None and increment_num is not None:
            # Find the increment meta to get total_time & fraction
            for inc in self.dat_parser.get_step_data(step_num):
                if inc.get('increment') == increment_num:
                    inc_meta = inc
                    break

        pressure_val = self._compute_effective_pressure(step_num, inc_meta)

        return {
            "surface_name": surface_name,
            "reference_point": reference_point,
            "num_faces": len(faces),
            "num_nodes": len(node_ids),
            "original_volume": original_volume,
            "displaced_volume": displaced_volume,
            "volume_change": volume_change,
            "percent_change": (volume_change / original_volume * 100.0) if (volume_change is not None and original_volume) else None,
            "pressure": pressure_val,
        }

    def analyze_volume_history(self, 
                               surface_name: str,
                               step_num: int,
                               reference_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                               verbose: bool = True) -> pd.DataFrame:
        _t0 = time.time()
        step_data = self.dat_parser.get_step_data(step_num)

        if not step_data:
            return pd.DataFrame()

        volume_history = []

        for inc_meta in step_data:
            inc_num = inc_meta.get('increment')

            vol_data = self.analyze_surface_volume(
                surface_name=surface_name,
                reference_point=reference_point,
                step_num=step_num,
                increment_num=inc_num
            )

            eff_p = self._compute_effective_pressure(step_num, inc_meta)
            vol_data['pressure'] = eff_p

            vol_data.update({
                'step': step_num,
                'increment': inc_num,
                'total_time': inc_meta.get('total_time'),
                'step_time': inc_meta.get('step_time'),
                'time_increment': inc_meta.get('time_increment'),
                'fraction_completed': inc_meta.get('fraction_completed'),
            })

            volume_history.append(vol_data)

        if verbose:
            print(f"Volume history analysis for step {step_num} took {time.time() - _t0:.2f} seconds.")
        return pd.DataFrame(volume_history)

    def get_step_boundaries(self, step_index: int = 0) -> Dict:
        return self.abaqus_input.get_step_boundaries(step_index)
    
    def get_step_loads(self, step_index: int = 0) -> List[Dict]:
        return self.abaqus_input.get_step_loads(step_index)
    
    def summary(self) -> Dict:
        dat_summary = self.dat_parser.summary()
        return {
            'input_file': self.abaqus_input.filename,
            'dat_file': self.dat_parser.dat_file,
            'node_sets': {
                'defined': list(self.abaqus_input.nsets.keys()),
                'with_output': dat_summary['discovered_node_sets']
            },
            'element_sets': list(self.abaqus_input.elsets.keys()),
            'surfaces': list(self.abaqus_input.surfaces.keys()),
            'num_nodes': len(self.abaqus_input.tabular_data.get('nodes', [])),
            'num_elements': len(self.abaqus_input.tabular_data.get('elements', [])),
            'steps': dat_summary['steps'],
            'boundary_conditions': {
                f'step_{i}': self.get_step_boundaries(i) 
                for i in range(len(self.abaqus_input.splitted_content.get('steps', [])))
            },
            'loads': {
                f'step_{i}': self.get_step_loads(i) 
                for i in range(len(self.abaqus_input.splitted_content.get('steps', [])))
            }
        }
