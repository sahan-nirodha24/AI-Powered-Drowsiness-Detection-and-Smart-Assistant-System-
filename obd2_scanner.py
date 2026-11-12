import obd
obd.logger.setLevel(obd.logging.DEBUG)
import pandas as pd
import time
import random
from datetime import datetime
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import serial.tools.list_ports
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import analysis_engine

class OBD2Scanner:
    def __init__(self):
        self.connection = None
        self.is_scanning = False
        self.data_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.scan_thread = None
        self.data_buffer = []
        self.buffer_size = 10  # Save to Firebase in batches or immediately
        self.session_id = None
        self.all_session_data = []  # Accumulate all data for local Excel saving (Stage 1)
        
        self.firebase_app = None
        self._initialize_firebase()
        
        # Common OBD2 PIDs to monitor  initialize as empty dictionary
        self.pids_to_monitor = {}
        
        # Safely add commands that exist in the current OBD library version
        self._initialize_available_commands()
        
        # Available PIDs 
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
                self.connection = obd.OBD(port, fast=False)
            else:
                # Auto-detect port manually to avoid OSError on specific invalid ports
                import serial.tools.list_ports
                ports = [comport.device for comport in serial.tools.list_ports.comports()]
                print(f"Available ports for Auto scan: {ports}")
                
                connected = False
                last_error = None
                for p in ports:
                    try:
                        print(f"Trying port {p}...")
                        self.connection = obd.OBD(p, fast=False)
                        if self.connection.is_connected():
                            print(f"Successfully connected to {p}")
                            connected = True
                            break
                        else:
                            self.connection.close()
                    except Exception as e:
                        print(f"Failed on {p}: {e}")
                        last_error = e
                
                if not connected:
                    if last_error:
                        raise Exception(f"Failed to scan ports. Last error: {last_error}")
                    else:
                        self.connection = obd.OBD(fast=False)
            
            if self.connection and self.connection.is_connected():
                print("Connection successful, discovering PIDs...")
                self.discover_available_pids()
                return True, f"Connected! Found {len(self.available_pids)} PIDs"
            else:
                return False, "Failed to connect to OBD2 adapter. Check device."
        except Exception as e:
            print(f"Connection error: {e}")
            return False, f"Connection failed. Please check if device is paired and car is ON. Details: {str(e)}"
    
    def discover_available_pids(self):
        """Discover which PIDs are available on the connected vehicle"""
        self.available_pids = {}
        if not self.connection or not self.connection.is_connected():
            return
            
        for name, command in self.pids_to_monitor.items():
            try:
                if self.connection.supports(command):
                    self.available_pids[name] = command
            except Exception as e:
                print(f"Error checking {name}: {e}")
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if already initialized to avoid errors
            if not firebase_admin._apps:
                # Replace with the path to your actual service account key JSON file
                # You must create this file from your Firebase console
                cred_path = "firebase_credentials.json"
                if os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                    self.firebase_app = firebase_admin.initialize_app(cred, {
                        'databaseURL': 'https://obd2-scanner-app-default-rtdb.firebaseio.com/' # UPDATE THIS URL
                    })
                    print("Firebase initialized successfully.")
                else:
                    print(f"Firebase credentials file not found at {cred_path}.")
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
    
    def start_scanning(self):
        """Start real-time data collection"""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.all_session_data = []  # Reset session data cache
        print(f"Starting Firebase session: {self.session_id}")
            
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
        self._push_to_firebase()
        # Save session to Excel locally (Stage 1)
        self._save_to_excel()
    
    def _scan_loop(self):
        """Main scanning loop"""
        last_error_check = 0
        while self.is_scanning:
            if not self.connection or not self.connection.is_connected():
                break
                
            try:
                # Periodically check errors in background to avoid freezing UI
                if time.time() - last_error_check > 5:
                    err_data = self.check_errors()
                    if err_data:
                        self.error_queue.put(err_data)
                    last_error_check = time.time()
                    
                timestamp = datetime.now()
                row_data = {'Timestamp': timestamp}
                
                # Query all available PIDs
                if not self.available_pids:
                    self.discover_available_pids()

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
                self.all_session_data.append(row_data)
                
                # Save to Firebase if buffer is full
                if len(self.data_buffer) >= self.buffer_size:
                    self._push_to_firebase()
                
                # Put data in queue for GUI updates
                self.data_queue.put(row_data)
                
                time.sleep(0.1)  # 10Hz sampling rate
                
            except Exception as e:
                print(f"Scanning error: {e}")
                time.sleep(1)
                
    def _push_to_firebase(self):
        """Save buffered data to Firebase"""
        if not self.data_buffer or not firebase_admin._apps or not self.session_id:
            return
        
        try:
            ref = db.reference(f'/obd2_data/{self.session_id}')
            for row in self.data_buffer:
                # Convert datetime to string for JSON serialization
                row_copy = row.copy()
                if 'Timestamp' in row_copy and isinstance(row_copy['Timestamp'], datetime):
                    row_copy['Timestamp'] = row_copy['Timestamp'].isoformat()
                
                # Push generates a unique key for each reading
                ref.push(row_copy)
                
            print(f"Pushed {len(self.data_buffer)} records to Firebase")
            self.data_buffer.clear()
            
        except Exception as e:
            print(f"Error pushing to Firebase: {e}")
            
    def _save_to_excel(self):
        """Save all session data to a local Excel file"""
        if not self.all_session_data:
            print("No data collected in this session to save to Excel.")
            return
            
        try:
            # Convert datetime objects to string for Excel compatibility
            processed_data = []
            for row in self.all_session_data:
                row_copy = row.copy()
                if 'Timestamp' in row_copy and isinstance(row_copy['Timestamp'], datetime):
                    row_copy['Timestamp'] = row_copy['Timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                processed_data.append(row_copy)
                
            df = pd.DataFrame(processed_data)
            
            # Ensure the output directory exists
            output_dir = "local_excel_data"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            filename = os.path.join(output_dir, f"{self.session_id}.xlsx")
            df.to_excel(filename, index=False)
            print(f"Session data successfully saved to local Excel: {filename}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            
    def check_errors(self):
        """Query DTCs and MIL status from the vehicle"""
        if not self.connection or not self.connection.is_connected():
            return None
            
        try:
            # Query DTCs
            dtc_response = self.connection.query(obd.commands.GET_DTC)
            dtcs = []
            if not dtc_response.is_null():
                for dtc in dtc_response.value:
                    dtcs.append({
                        'code': dtc[0],
                        'description': dtc[1],
                        'type': 'Powertrain' if dtc[0].startswith('P') else 'Network' if dtc[0].startswith('U') else 'Body' if dtc[0].startswith('B') else 'Chassis',
                        'status': 'Active',
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            # For demonstration, generate some mock data if empty and scanning
            if not dtcs and self.is_scanning:
                if random.random() > 0.95:
                    dtcs.append({
                        'code': 'P0301',
                        'description': 'Cylinder 1 misfire detected',
                        'type': 'Powertrain',
                        'status': 'Active',
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            # Push errors to firebase
            if dtcs and self.session_id and firebase_admin._apps:
                ref = db.reference(f'/obd2_errors/{self.session_id}')
                for dtc in dtcs:
                    ref.push(dtc)
                    
            return {
                'active_count': len(dtcs),
                'pending_count': 0, # Placeholder
                'permanent_count': 0, # Placeholder
                'mil_on': len(dtcs) > 0,
                'errors': dtcs
            }
        except Exception as e:
            print(f"Error checking DTCs: {e}")
            return None
    
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
        self.analytics = analysis_engine.AnalyticsEngine()
        self.update_thread = None
        
        self.create_widgets()
        self.update_gui()
        
        # Populate ports initially
        self.refresh_ports()
    
    def create_widgets(self):
        # Create Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Scanner Tab
        self.scanner_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.scanner_tab, text="Scanner")
        self.create_scanner_widgets(self.scanner_tab)
        
        # 2. Error Monitor Tab
        self.error_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.error_tab, text="Error Monitor")
        self.create_error_widgets(self.error_tab)
        
        # 3. Analytics Tab
        self.analytics_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.create_analytics_widgets(self.analytics_tab)

    def create_scanner_widgets(self, parent):
        # Connection frame
        conn_frame = ttk.LabelFrame(parent, text="Connection", padding="10")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, sticky=tk.W)
        
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=20)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        ttk.Button(conn_frame, text="⟳", width=3, command=self.refresh_ports).grid(row=0, column=2, padx=2)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect)
        self.connect_btn.grid(row=0, column=3, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=self.disconnect)
        self.disconnect_btn.grid(row=0, column=4, padx=5)
        self.disconnect_btn.config(state='disabled')
        
        self.status_var = tk.StringVar(value="Not connected. Select a port and click Connect.")
        status_lbl = ttk.Label(conn_frame, textvariable=self.status_var, wraplength=500)
        status_lbl.grid(row=1, column=0, columnspan=5, pady=5, sticky=tk.W)
        
        # Control frame
        ctrl_frame = ttk.LabelFrame(parent, text="Control", padding="10")
        ctrl_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.start_btn = ttk.Button(ctrl_frame, text="Start Scanning", command=self.start_scan)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.start_btn.config(state='disabled')
        
        self.stop_btn = ttk.Button(ctrl_frame, text="Stop Scanning", command=self.stop_scan)
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.stop_btn.config(state='disabled')
        
        # Data display frame
        data_frame = ttk.LabelFrame(parent, text="Real-time Data", padding="10")
        data_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        columns = ('Parameter', 'Value', 'Unit')
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)

    def create_error_widgets(self, parent):
        # Top summary cards frame
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Helper to create a card
        def create_card(parent_frame, title, value_var, color):
            frame = tk.Frame(parent_frame, bg='#f8f9fa', padx=15, pady=10, relief=tk.FLAT, bd=1)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(frame, text=title, bg='#f8f9fa', fg='#6c757d', font=('Helvetica', 10)).pack(anchor='w')
            tk.Label(frame, textvariable=value_var, bg='#f8f9fa', fg=color, font=('Helvetica', 20, 'bold')).pack(anchor='w', pady=(5, 0))
            return frame

        self.active_dtc_var = tk.StringVar(value="0")
        self.pending_dtc_var = tk.StringVar(value="0")
        self.permanent_dtc_var = tk.StringVar(value="0")
        self.mil_var = tk.StringVar(value="OFF")

        create_card(cards_frame, "Active DTCs", self.active_dtc_var, "#dc3545")
        create_card(cards_frame, "Pending DTCs", self.pending_dtc_var, "#856404")
        create_card(cards_frame, "Permanent DTCs", self.permanent_dtc_var, "#28a745")
        create_card(cards_frame, "MIL (check engine)", self.mil_var, "#dc3545")
        
        # Tabs for error types
        error_notebook = ttk.Notebook(parent)
        error_notebook.pack(fill=tk.BOTH, expand=True)
        
        active_tab = ttk.Frame(error_notebook)
        error_notebook.add(active_tab, text="Active errors")
        
        # Treeview for errors
        columns = ('Code', 'Type', 'Description', 'First seen', 'Saved to Firebase')
        self.error_tree = ttk.Treeview(active_tab, columns=columns, show='headings', height=10)
        
        self.error_tree.heading('Code', text='Code')
        self.error_tree.column('Code', width=80)
        self.error_tree.heading('Type', text='Type')
        self.error_tree.column('Type', width=100)
        self.error_tree.heading('Description', text='Description')
        self.error_tree.column('Description', width=300)
        self.error_tree.heading('First seen', text='First seen')
        self.error_tree.column('First seen', width=100)
        self.error_tree.heading('Saved to Firebase', text='Saved to Firebase')
        self.error_tree.column('Saved to Firebase', width=120)
        
        self.error_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom status
        self.error_status_var = tk.StringVar(value="Not scanning.")
        ttk.Label(parent, textvariable=self.error_status_var, font=('Helvetica', 9)).pack(anchor='w', pady=(10, 0))
        
    def create_analytics_widgets(self, parent):
        # Top frame for controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Session ID:").pack(side=tk.LEFT, padx=5)
        self.session_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.session_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Parameter:").pack(side=tk.LEFT, padx=5)
        self.param_var = tk.StringVar(value="RPM")
        ttk.Combobox(control_frame, textvariable=self.param_var, values=["RPM", "SPEED", "ENGINE_LOAD", "COOLANT_TEMP"]).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        
        # Frame for plots
        self.plot_frame = ttk.Frame(parent)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def run_analysis(self):
        session_id = self.session_var.get() or self.scanner.session_id
        if not session_id:
            messagebox.showerror("Error", "No session ID available. Start a scan first or enter an ID.")
            return
            
        param = self.param_var.get()
        self.session_var.set(session_id) # Update UI if we used active session
        
        # Run in background thread
        threading.Thread(target=self._analysis_thread, args=(session_id, param), daemon=True).start()
        
    def _analysis_thread(self, session_id, param):
        df = self.analytics.load_session_from_firebase(session_id)
        if df is None or param not in df.columns:
            self.root.after(0, lambda: messagebox.showerror("Error", "Could not load data or parameter not found."))
            return
            
        # Process
        df = self.analytics.apply_smoothing(df, param)
        df = self.analytics.detect_anomalies(df, param)
        future_times, forecast = self.analytics.forecast_trend(df, param)
        
        # Update UI
        self.root.after(0, lambda: self._update_plot(df, param, future_times, forecast))
        
    def _update_plot(self, df, param, future_times, forecast):
        # Clear old plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        
        # Trend chart with smoothing
        ax1.plot(df.index, df[param], label='Raw', alpha=0.5)
        if f'{param}_EMA' in df.columns:
            ax1.plot(df.index, df[f'{param}_EMA'], label='EMA', color='orange', linewidth=2)
        ax1.set_title(f"{param} Trend & Smoothing")
        ax1.legend()
        
        # Anomaly and Forecast chart
        ax2.plot(df.index, df[param], label='Raw', color='blue')
        # Plot anomalies
        if f'{param}_Anomaly' in df.columns:
            anomalies = df[df[f'{param}_Anomaly'] == True]
            if not anomalies.empty:
                ax2.scatter(anomalies.index, anomalies[param], color='red', label='Anomaly', zorder=5)
            
        # Plot forecast
        if future_times is not None and forecast is not None:
            ax2.plot(future_times, forecast, label='Forecast', color='green', linestyle='--')
            
        ax2.set_title("Anomaly Detection & Forecast")
        ax2.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
    
    def start_scan(self):
        self.scanner.start_scanning()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set(f"Scanning... Data is being pushed to Firebase (Session: {self.scanner.session_id}).")
    
    def stop_scan(self):
        self.scanner.stop_scanning()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        excel_path = f"local_excel_data\\{self.scanner.session_id}.xlsx"
        self.status_var.set(f"Scanning stopped. Data saved to Firebase & local Excel: {excel_path}")
    
    def update_gui(self):
        # Process all pending data in the queue
        try:
            while True:
                # Get data without blocking
                data = self.scanner.data_queue.get_nowait()
                self.update_data_display(data)
        except queue.Empty:
            pass
            
        # Process all pending error data
        try:
            while True:
                err_data = self.scanner.error_queue.get_nowait()
                self._update_errors_ui(err_data)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_gui)
        
    def _update_errors_ui(self, error_data):
        if error_data:
            self.active_dtc_var.set(str(error_data['active_count']))
            self.pending_dtc_var.set(str(error_data['pending_count']))
            self.permanent_dtc_var.set(str(error_data['permanent_count']))
            self.mil_var.set("ON" if error_data['mil_on'] else "OFF")
            self.mil_var.set("ON" if error_data['mil_on'] else "OFF")
            
            # Clear tree
            for item in self.error_tree.get_children():
                self.error_tree.delete(item)
                
            # Populate tree
            for error in error_data['errors']:
                self.error_tree.insert('', 'end', values=(
                    error['code'],
                    error['type'],
                    error['description'],
                    error['timestamp'],
                    "Saved" if self.scanner.session_id else "No"
                ))
            
            if self.scanner.session_id:
                self.error_status_var.set(f"Firebase connected — errors saved to /obd2_errors/{self.scanner.session_id}/")
            else:
                self.error_status_var.set("Not connected to Firebase.")
    
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