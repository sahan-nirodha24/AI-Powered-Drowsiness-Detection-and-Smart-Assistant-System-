import obd
import pandas as pd
import time
from datetime import datetime
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import serial.tools.list_ports

class OBD2Scanner:
    def __init__(self):
        self.connection = None
        self.is_scanning = False
        self.data_queue = queue.Queue()
        self.excel_file = None
        self.scan_thread = None
        self.data_buffer = []
        self.buffer_size = 10  # Save to Excel every 10 readings
        
        # Default Excel file path
        self.default_excel_path = os.path.join(os.path.expanduser("~"), "Desktop", "OBD2_Data.xlsx")
        
        # Common OBD2 PIDs to monitor - initialize as empty dictionary
        self.pids_to_monitor = {}
        
        # Safely add commands that exist in the current OBD library version
        self._initialize_available_commands()
        
        # Available PIDs (will be populated after connection)
        self.available_pids = {}
        
    def _initialize_available_commands(self):
        """Initialize only the commands that are available in the current OBD library version"""
        # List of potential commands to check
        potential_commands = [
            'ENGINE_LOAD',
            'COOLANT_TEMP',
            'SHORT_FUEL_TRIM_1',
            'LONG_FUEL_TRIM_1',
            'INTAKE_PRESSURE',
            'RPM',
            'SPEED',
            'TIMING_ADVANCE',
            'INTAKE_TEMP',
            'MAF',
            'THROTTLE_POS',
            'O2_B1S1',
            'O2_B1S2',
            'FUEL_LEVEL',
            'BAROMETRIC_PRESSURE',
            'CATALYST_TEMP_B1S1',
            'CONTROL_MODULE_VOLTAGE',
            'ABSOLUTE_LOAD',
            'FUEL_RAIL_PRESSURE_VAC',
            'FUEL_RAIL_PRESSURE_DIRECT',
            'COMMANDED_EGR',
            'EGR_ERROR',
            'EVAP_VAPOR_PRESSURE',
            'FUEL_INJECT_TIMING',
            'ENGINE_FUEL_RATE',
        ]
        
        # Only add commands that actually exist in the current library
        for command_name in potential_commands:
            if hasattr(obd.commands, command_name):
                self.pids_to_monitor[command_name] = getattr(obd.commands, command_name)
            else:
                pass
        
    def connect_bluetooth(self, port=None):
        """Connect to OBD2 adapter via Bluetooth"""
        try:
            print(f"Attempting connection on port: {port if port else 'Auto'}")
            if port and port != "Auto":
                self.connection = obd.OBD(port)
            else:
                # Auto-detect port
                self.connection = obd.OBD()
            
            if self.connection.is_connected():
                print("Connection successful, discovering PIDs...")
                self.discover_available_pids()
                return True, f"Connected! Found {len(self.available_pids)} PIDs"
            else:
                return False, "Failed to connect to OBD2 adapter. Check device."
        except Exception as e:
            print(f"Connection error: {e}")
            return False, str(e)
    
    def discover_available_pids(self):
        """Discover which PIDs are available on the connected vehicle"""
        self.available_pids = {}
        
        for name, command in self.pids_to_monitor.items():
            try:
                # Use a lightweight check or just query
                response = self.connection.query(command)
                if not response.is_null():
                    self.available_pids[name] = command
            except Exception as e:
                print(f"Error checking {name}: {e}")
    
    def set_excel_path(self, custom_path=None):
        """Set custom Excel file path"""
        if custom_path:
            self.excel_file = custom_path
        else:
            # Auto-generate path if none provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"OBD2_Data_{timestamp}.xlsx"
            self.excel_file = os.path.join(os.path.expanduser("~"), "Desktop", filename)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(self.excel_file)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception:
                pass # Ignore if drive root or similar
    
    def start_scanning(self, excel_file_path=None):
        """Start real-time data collection"""
        if excel_file_path:
            self.excel_file = excel_file_path
        elif not self.excel_file:
            self.set_excel_path()
            
        directory = os.path.dirname(self.excel_file)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception:
                pass
            
        self.is_scanning = True
        self.scan_thread = threading.Thread(target=self._scan_loop)
        self.scan_thread.daemon = True
        self.scan_thread.start()
    
    def stop_scanning(self):
        """Stop data collection"""
        self.is_scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=1.0)
        # Save any remaining data
        self._save_buffer_to_excel()
    
    def _scan_loop(self):
        """Main scanning loop"""
        while self.is_scanning and self.connection and self.connection.is_connected():
            try:
                timestamp = datetime.now()
                row_data = {'Timestamp': timestamp}
                
                # Query all available PIDs
                if not self.available_pids:
                    # Retry discovery if empty (sometimes happens on weak connection)
                    for name, command in self.pids_to_monitor.items():
                        self.available_pids[name] = command

                for name, command in self.available_pids.items():
                    try:
                        response = self.connection.query(command)
                        if not response.is_null():
                            # Extract numeric value
                            value = response.value
                            if hasattr(value, 'magnitude'):
                                value = value.magnitude
                            row_data[name] = value
                        else:
                            row_data[name] = None
                    except Exception as e:
                        row_data[name] = None
                
                # Add to buffer
                self.data_buffer.append(row_data)
                
                # Save to Excel if buffer is full
                if len(self.data_buffer) >= self.buffer_size:
                    self._save_buffer_to_excel()
                
                # Put data in queue for GUI updates
                self.data_queue.put(row_data)
                
                time.sleep(0.1)  # 10Hz sampling rate
                
            except Exception as e:
                print(f"Scanning error: {e}")
                time.sleep(1)
                
    def _save_buffer_to_excel(self):
        """Save buffered data to Excel file"""
        if not self.data_buffer or not self.excel_file:
            return
        
        try:
            df = pd.DataFrame(self.data_buffer)
            
            # Check if file exists
            if os.path.exists(self.excel_file):
                # Append to existing file
                try:
                    # Read existing data to get the next row
                    try:
                        existing_df = pd.read_excel(self.excel_file, sheet_name='OBD2_Data')
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                    except ValueError:
                        # Sheet might not exist
                        combined_df = df
                        
                    combined_df.to_excel(self.excel_file, sheet_name='OBD2_Data', index=False)
                except Exception as e:
                    # If there's an issue with appending, just overwrite/create
                    print(f"Append error, creating new file: {e}")
                    df.to_excel(self.excel_file, sheet_name='OBD2_Data', index=False)
            else:
                # Create new file
                df.to_excel(self.excel_file, sheet_name='OBD2_Data', index=False)
            
            self.data_buffer.clear()
            print(f"Saved {len(df)} records to Excel")
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
    
    def disconnect(self):
        """Disconnect from OBD2 adapter"""
        self.stop_scanning()
        if self.connection:
            self.connection.close()
            self.connection = None

class OBD2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern OBD2 Scanner - v2.0")
        self.root.geometry("900x700")
        
        self.scanner = OBD2Scanner()
        self.update_thread = None
        
        self.create_widgets()
        self.update_gui()
        
        # Populate ports initially
        self.refresh_ports()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection", padding="10")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, sticky=tk.W)
        
        # Port Selection Combobox
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=20)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        # Refresh Ports Button
        ttk.Button(conn_frame, text="⟳", width=3, command=self.refresh_ports).grid(row=0, column=2, padx=2)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect)
        self.connect_btn.grid(row=0, column=3, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=self.disconnect)
        self.disconnect_btn.grid(row=0, column=4, padx=5)
        self.disconnect_btn.config(state='disabled')
        
        # Status Label
        self.status_var = tk.StringVar(value="Not connected. Select a port and click Connect.")
        status_lbl = ttk.Label(conn_frame, textvariable=self.status_var, wraplength=500)
        status_lbl.grid(row=1, column=0, columnspan=5, pady=5, sticky=tk.W)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Excel File Path", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Excel file path
        self.file_var = tk.StringVar(value=self.scanner.default_excel_path)
        ttk.Entry(file_frame, textvariable=self.file_var, width=60).grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=5)
        
        # Quick path buttons
        quick_frame = ttk.Frame(file_frame)
        quick_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(quick_frame, text="Desktop", command=self.set_desktop_path).grid(row=0, column=0, padx=2)
        ttk.Button(quick_frame, text="Documents", command=self.set_documents_path).grid(row=0, column=1, padx=2)
        ttk.Button(quick_frame, text="Current Folder", command=self.set_current_path).grid(row=0, column=2, padx=2)
        ttk.Button(quick_frame, text="Auto-Generate", command=self.auto_generate_path).grid(row=0, column=3, padx=2)
        
        # Control frame
        ctrl_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        ctrl_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.start_btn = ttk.Button(ctrl_frame, text="Start Scanning", command=self.start_scan)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.start_btn.config(state='disabled')
        
        self.stop_btn = ttk.Button(ctrl_frame, text="Stop Scanning", command=self.stop_scan)
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.stop_btn.config(state='disabled')
        
        # Data display frame
        data_frame = ttk.LabelFrame(main_frame, text="Real-time Data", padding="10")
        data_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create treeview for data display
        columns = ('Parameter', 'Value', 'Unit')
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)

    def refresh_ports(self):
        """List available COM ports"""
        try:
            ports = [comport.device for comport in serial.tools.list_ports.comports()]
        except Exception:
            ports = []
        ports.insert(0, "Auto") # Add Auto option
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)
    
    def set_desktop_path(self):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"OBD2_Data_{timestamp}.xlsx"
        self.file_var.set(os.path.join(desktop_path, filename))
    
    def set_documents_path(self):
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"OBD2_Data_{timestamp}.xlsx"
        self.file_var.set(os.path.join(documents_path, filename))
    
    def set_current_path(self):
        current_path = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"OBD2_Data_{timestamp}.xlsx"
        self.file_var.set(os.path.join(current_path, filename))
    
    def auto_generate_path(self):
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        obd_folder = os.path.join(documents_path, "OBD2_Data")
        
        if not os.path.exists(obd_folder):
            try:
                os.makedirs(obd_folder)
            except Exception:
                pass
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CarData_{timestamp}.xlsx"
        self.file_var.set(os.path.join(obd_folder, filename))
    
    def connect(self):
        """Initiate connection in a background thread"""
        port = self.port_var.get()
        if not port:
            port = "Auto"
            
        self.connect_btn.config(state='disabled')
        self.port_combo.config(state='disabled')
        self.status_var.set("Connecting... This may take a moment...")
        
        # Start connection thread
        threading.Thread(target=self._connect_thread, args=(port,), daemon=True).start()
        
    def _connect_thread(self, port):
        """Background connection task"""
        success, message = self.scanner.connect_bluetooth(port)
        # Schedule UI update in main thread
        self.root.after(0, lambda: self._on_connect_completed(success, message))
        
    def _on_connect_completed(self, success, message):
        """Handle connection result on main thread"""
        self.status_var.set(message)
        if success:
            self.disconnect_btn.config(state='normal')
            self.start_btn.config(state='normal')
            messagebox.showinfo("Success", message)
        else:
            self.connect_btn.config(state='normal')
            self.port_combo.config(state='normal')
            messagebox.showerror("Connection Failed", message)
    
    def disconnect(self):
        self.scanner.disconnect()
        self.status_var.set("Disconnected")
        self.connect_btn.config(state='normal')
        self.disconnect_btn.config(state='disabled')
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
        self.port_combo.config(state='normal')
    
    def browse_file(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
    
    def start_scan(self):
        if not self.file_var.get():
            messagebox.showerror("Error", "Please select an Excel file path")
            return
        
        self.scanner.start_scanning(self.file_var.get())
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("Scanning... Data is being saved to Excel.")
    
    def stop_scan(self):
        self.scanner.stop_scanning()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Scanning stopped. Data saved.")
    
    def update_gui(self):
        # Process all pending data in the queue
        try:
            while True:
                # Get data without blocking
                data = self.scanner.data_queue.get_nowait()
                self.update_data_display(data)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def update_data_display(self, data):
        # We can implement a smarter update here to avoid flicker if needed
        # For now, full refresh is okay for small item counts
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for key, value in data.items():
            if key != 'Timestamp':
                unit = self.get_unit(key)
                if isinstance(value, float):
                    display_value = f"{value:.2f}"
                elif value is None:
                    display_value = "No Data"
                else:
                    display_value = str(value)
                    
                self.tree.insert('', 'end', values=(key, display_value, unit))
    
    def get_unit(self, parameter):
        units = {
            'ENGINE_LOAD': '%',
            'COOLANT_TEMP': '°C',
            'SHORT_FUEL_TRIM_1': '%',
            'LONG_FUEL_TRIM_1': '%',
            'INTAKE_PRESSURE': 'kPa',
            'RPM': 'RPM',
            'SPEED': 'km/h',
            'TIMING_ADVANCE': '°',
            'INTAKE_TEMP': '°C',
            'MAF': 'g/s',
            'THROTTLE_POS': '%',
            'O2_B1S1': 'V',
            'O2_B1S2': 'V',
            'FUEL_LEVEL': '%',
            'BAROMETRIC_PRESSURE': 'kPa',
            'CATALYST_TEMP_B1S1': '°C',
            'CONTROL_MODULE_VOLTAGE': 'V',
            'ABSOLUTE_LOAD': '%',
            'FUEL_RAIL_PRESSURE_VAC': 'kPa',
            'FUEL_RAIL_PRESSURE_DIRECT': 'kPa',
            'COMMANDED_EGR': '%',
            'EGR_ERROR': '%',
            'EVAP_VAPOR_PRESSURE': 'Pa',
            'FUEL_INJECT_TIMING': '°',
            'ENGINE_FUEL_RATE': 'L/h',
        }
        return units.get(parameter, '')

def main():
    root = tk.Tk()
    app = OBD2GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()