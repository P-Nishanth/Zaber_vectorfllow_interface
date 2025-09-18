# Universe.py
""""
Author: Nish
Supervisor: Dries
Sydney Propulsion Laboratory
"""

import pandas as pd
import numpy as np
from zaber_motion.ascii import Connection
from zaber_motion import Units
import time
from datetime import datetime, timedelta
import threading
import serial
import pyvista as pv
from scipy.interpolate import griddata
from tqdm import tqdm

def rotate_device(device_index=3, angle=0, speed_dps=10):
    """
    Rotate a Zaber rotary axis to a specified angle at a given speed (degrees/sec).

    Limits:
        max speed = 24 deg/s
        min speed = 0.00095 deg/s
        angle range = 0–360 degrees (update if needed)
    """
    SPEED_MIN = 0.00095
    SPEED_MAX = 24.0
    ANGLE_MIN = 0
    ANGLE_MAX = 360

    if not (SPEED_MIN <= speed_dps <= SPEED_MAX):
        raise ValueError(f"Speed {speed_dps}°/s out of bounds ({SPEED_MIN}–{SPEED_MAX})")
    if not (ANGLE_MIN <= angle <= ANGLE_MAX):
        raise ValueError(f"Angle {angle}° out of bounds ({ANGLE_MIN}–{ANGLE_MAX})")

    with Connection.open_serial_port("COM5") as connection:
        device = connection.detect_devices()[device_index]
        axis = device.get_axis(1)

        if not axis.is_homed():
            print(f"Homing rotary axis (device {device.serial_number})...")
            axis.home()
            time.sleep(1)

        from zaber_motion import Units

        axis.settings.set("maxspeed", speed_dps, unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
        print(f"Set rotary maxspeed to {speed_dps} deg/s")

        print(f"Rotating to {angle}°...")
        axis.move_absolute(angle, Units.ANGLE_DEGREES)

        print(f"Motor Rotated to {angle}°")


def move_probe_to(x_mm=0, y_mm=0, z_mm=0, speed_xy=25, speed_z=13, ports="COM5"):
    """
        Move individual X, Y, Z stages with speed and travel range checks.

        X & Y axis:
            max speed = 53 mm/s
            min speed = 6.1e-5 mm/s
            travel = 300 mm
        Z axis:
            max speed = 26 mm/s
            min speed = 0.000029 mm/s
            travel = 203.2 mm
    """
    SPEED_XY_MIN, SPEED_XY_MAX = 6.1e-5, 53
    SPEED_Z_MIN, SPEED_Z_MAX = 0.000029, 26
    RANGE_X = 300
    RANGE_Y = 300
    RANGE_Z = 203.2

    if not (0 <= x_mm <= RANGE_X):
        raise ValueError(f"X position {x_mm} mm out of range (0–{RANGE_X})")
    if not (0 <= y_mm <= RANGE_Y):
        raise ValueError(f"Y position {y_mm} mm out of range (0–{RANGE_Y})")
    if not (0 <= z_mm <= RANGE_Z):
        raise ValueError(f"Z position {z_mm} mm out of range (0–{RANGE_Z})")
    if not (SPEED_XY_MIN <= speed_xy <= SPEED_XY_MAX):
        raise ValueError(f"XY speed {speed_xy} mm/s out of range ({SPEED_XY_MIN}–{SPEED_XY_MAX})")
    if not (SPEED_Z_MIN <= speed_z <= SPEED_Z_MAX):
        raise ValueError(f"Z speed {speed_z} mm/s out of range ({SPEED_Z_MIN}–{SPEED_Z_MAX})")

    with Connection.open_serial_port(ports) as connection:
        device_list = connection.detect_devices()

        # X axis
        x_axis = device_list[0].get_axis(1)
        if not x_axis.is_homed(): x_axis.home()
        x_axis.settings.set("maxspeed", speed_xy, unit=Units.VELOCITY_MILLIMETRES_PER_SECOND)
        x_axis.move_absolute(x_mm, Units.LENGTH_MILLIMETRES)

        # Y axis
        y_axis = device_list[1].get_axis(1)
        if not y_axis.is_homed(): y_axis.home()
        y_axis.settings.set("maxspeed", speed_xy, unit=Units.VELOCITY_MILLIMETRES_PER_SECOND)
        y_axis.move_absolute(y_mm, Units.LENGTH_MILLIMETRES)

        # Z axis
        z_axis = device_list[2].get_axis(1)
        if not z_axis.is_homed(): z_axis.home()
        z_axis.settings.set("maxspeed", speed_z, unit=Units.VELOCITY_MILLIMETRES_PER_SECOND)
        z_axis.move_absolute(z_mm, Units.LENGTH_MILLIMETRES)

        print(f"Probe moved to X={x_mm} mm, Y={y_mm} mm, Z={z_mm} mm.")


def perform_volume_sweep(
        x_bounds=(0, 300), y_bounds=(0, 295), z_bounds=(0, 203.2),
        num_steps_x=10, num_steps_y=5, num_steps_z=5,
        speed_x=20, speed_y=20, speed_z=10,
        port="COM5"
) -> pd.DataFrame:
    """
    Perform a volume sweep with a 3-axis Zaber traverse system.

    Parameters:
        x_bounds, y_bounds, z_bounds: (min, max) positions in mm
        num_steps_x, y, z: Number of points in each direction
        speed_x, speed_y, speed_z: Speeds in mm/s
        port: Serial port (e.g., /dev/ttyUSB0)

    Returns:
        DataFrame with (time, x, y, z) for each sweep location
    """
    # Generate sweep paths
    x_range = np.linspace(x_bounds[0], x_bounds[1], num_steps_x)
    y_range = np.linspace(y_bounds[1], y_bounds[0], num_steps_y)  # y: max to min
    z_range = np.linspace(z_bounds[1], z_bounds[0], num_steps_z)  # z: max to min

    # --- TQDM SETUP ---
    total_points = num_steps_x * num_steps_y * num_steps_z

    log = []
    start_time = time.time()

    with Connection.open_serial_port(port) as connection:
        devices = connection.detect_devices()
        x_axis = devices[0].get_axis(1)
        y_axis = devices[1].get_axis(1)
        z_axis = devices[2].get_axis(1)

        # Home axes if needed
        for axis, name in zip([x_axis, y_axis, z_axis], ["X", "Y", "Z"]):
            if not axis.is_homed():
                print(f"Homing {name} axis...")
                axis.home()

        # Set speeds
        x_axis.settings.set("maxspeed", speed_x, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        y_axis.settings.set("maxspeed", speed_y, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        z_axis.settings.set("maxspeed", speed_z, Units.VELOCITY_MILLIMETRES_PER_SECOND)

        # Initial move to (x_min, y_max, z_max) and wait
        x_axis.move_absolute(float(x_range[0]), Units.LENGTH_MILLIMETRES, wait_until_idle=False)
        y_axis.move_absolute(float(y_range[0]), Units.LENGTH_MILLIMETRES, wait_until_idle=False)
        z_axis.move_absolute(float(z_range[0]), Units.LENGTH_MILLIMETRES, wait_until_idle=False)
        for axis in [x_axis, y_axis, z_axis]:
            axis.wait_until_idle()

        # --- WRAP LOOPS WITH TQDM ---
        with tqdm(total=total_points, unit="point", desc="Volume Sweep") as pbar:
            for y in y_range:
                forward = True
                for z in z_range:
                    x_path = x_range if forward else reversed(x_range)
                    for x in x_path:
                        # Moves are blocking by default, ensuring we wait at each point
                        x_axis.move_absolute(float(x), Units.LENGTH_MILLIMETRES)
                        y_axis.move_absolute(float(y), Units.LENGTH_MILLIMETRES)
                        z_axis.move_absolute(float(z), Units.LENGTH_MILLIMETRES)

                        # Log position
                        log.append({
                            "timestamp": time.time() - start_time,
                            "x_mm": float(x),
                            "y_mm": float(y),
                            "z_mm": float(z)
                        })

                        # --- UPDATE PBAR ---
                        pbar.update(1)

                    forward = not forward  # Zig-zag

    df = pd.DataFrame(log)
    print("Volume sweep complete.")
    return df


def poll_positions_25hz(port="/dev/ttyUSB0") -> pd.DataFrame:
    """
    Poll positions at 25Hz during X-axis motion from 0 to 300mm.
    """
    poll_interval = 1 / 25  # 25Hz
    log = []

    with Connection.open_serial_port(port) as connection:
        devices = connection.detect_devices()
        x_axis = devices[0].get_axis(1)
        y_axis = devices[1].get_axis(1)
        z_axis = devices[2].get_axis(1)

        # Home axes if needed
        for axis, name in zip([x_axis, y_axis, z_axis], ["X", "Y", "Z"]):
            if not axis.is_homed():
                print(f"Homing {name} axis...")
                axis.home()

        # Move to known start position
        print("Positioning to start...")
        y_axis.move_absolute(290, Units.LENGTH_MILLIMETRES)
        z_axis.move_absolute(100, Units.LENGTH_MILLIMETRES)
        x_axis.move_absolute(0, Units.LENGTH_MILLIMETRES)
        time.sleep(1)

        # Start motion
        print("Starting X-axis move to 300mm...")
        motion_thread = threading.Thread(
            target=lambda: x_axis.move_absolute(300, Units.LENGTH_MILLIMETRES)
        )
        motion_thread.start()

        # Start timing now
        start = time.time()

        while motion_thread.is_alive():
            now = time.time()
            timestamp = now - start

            x = x_axis.get_position(Units.LENGTH_MILLIMETRES)
            y = y_axis.get_position(Units.LENGTH_MILLIMETRES)
            z = z_axis.get_position(Units.LENGTH_MILLIMETRES)

            log.append({
                "timestamp": timestamp,
                "x_mm": x,
                "y_mm": y,
                "z_mm": z
            })

            print(f"[{timestamp:.2f}s] x={x:.2f}, y={y:.2f}, z={z:.2f}")
            time.sleep(max(0, poll_interval - (time.time() - now)))

        motion_thread.join()

    print(f"Number of logged entries: {len(log)}")
    df = pd.DataFrame(log)
    print("25Hz polling complete.")
    return df


def perform_volume_sweep_with_polling(
    x_bounds=(0, 300), y_bounds=(0, 295), z_bounds=(0, 203.2),
    num_steps_x=10, num_steps_y=5, num_steps_z=5,
    speed_x=20, speed_y=20, speed_z=10,
    poll_frequency=50,
    port="/dev/ttyUSB0"
) -> pd.DataFrame:
    """
    Perform a 3D volume sweep with polling during each motion.

    Returns a DataFrame with timestamped (x, y, z) positions sampled at specified frequency.
    """
    x_range = np.linspace(x_bounds[0], x_bounds[1], num_steps_x)
    y_range = np.linspace(y_bounds[1], y_bounds[0], num_steps_y)  # y: max to min
    z_range = np.linspace(z_bounds[1], z_bounds[0], num_steps_z)  # z: max to min
    poll_interval = 1.0 / poll_frequency

    log = []

    with Connection.open_serial_port(port) as connection:
        devices = connection.detect_devices()
        x_axis = devices[0].get_axis(1)
        y_axis = devices[1].get_axis(1)
        z_axis = devices[2].get_axis(1)

        # Home axes if needed
        for axis, name in zip([x_axis, y_axis, z_axis], ["X", "Y", "Z"]):
            if not axis.is_homed():
                print(f"Homing {name} axis...")
                axis.home()

        # Set speeds
        x_axis.settings.set("maxspeed", speed_x, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        y_axis.settings.set("maxspeed", speed_y, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        z_axis.settings.set("maxspeed", speed_z, Units.VELOCITY_MILLIMETRES_PER_SECOND)

        # Move to starting corner
        x_axis.move_absolute(x_range[0], Units.LENGTH_MILLIMETRES)
        y_axis.move_absolute(y_range[0], Units.LENGTH_MILLIMETRES)
        z_axis.move_absolute(z_range[0], Units.LENGTH_MILLIMETRES)

        start = time.time()

        for y in y_range:
            forward = True
            for z in z_range:
                x_path = x_range if forward else reversed(x_range)
                for x in x_path:
                    # Start threads to move all axes
                    threads = []
                    if x_axis.get_position(Units.LENGTH_MILLIMETRES) != x:
                        threads.append(threading.Thread(
                            target=lambda: x_axis.move_absolute(x, Units.LENGTH_MILLIMETRES)
                        ))
                    if y_axis.get_position(Units.LENGTH_MILLIMETRES) != y:
                        threads.append(threading.Thread(
                            target=lambda: y_axis.move_absolute(y, Units.LENGTH_MILLIMETRES)
                        ))
                    if z_axis.get_position(Units.LENGTH_MILLIMETRES) != z:
                        threads.append(threading.Thread(
                            target=lambda: z_axis.move_absolute(z, Units.LENGTH_MILLIMETRES)
                        ))

                    for t in threads:
                        t.start()

                    # Poll while threads are alive
                    while any(t.is_alive() for t in threads):
                        now = time.time()
                        log.append({
                            "timestamp": now - start,
                            "x_mm": x_axis.get_position(Units.LENGTH_MILLIMETRES),
                            "y_mm": y_axis.get_position(Units.LENGTH_MILLIMETRES),
                            "z_mm": z_axis.get_position(Units.LENGTH_MILLIMETRES)
                        })
                        time.sleep(poll_interval)

                    for t in threads:
                        t.join()

                forward = not forward  # Zigzag

    df = pd.DataFrame(log)
    print(f"Sweep complete. {len(df)} samples collected.")
    return df

def get_probe_data(port='COM6', NUM_SAMPLES=100) -> pd.DataFrame:
    BAUDRATE = 115200
    TIMEOUT = 1  # seconds

    HEADERS = [
        # Group 2: Thermodynamic Inputs (14 pressures)
        "P1[Pa]", "P2[Pa]", "P3[Pa]", "P4[Pa]", "P5[Pa]", "P6[Pa]", "P7[Pa]", "P8[Pa]",
        "P9[Pa]", "P10[Pa]", "P11[Pa]", "P12[Pa]", "P13[Pa]", "P14[Pa]",
        "Pabs[Pa]", "Ttc[C]",

        # Group 3: Attitude
        "Theta[Deg]", "Phi[Deg]", "Alpha[Deg]", "Beta[Deg]",

        # Group 4: Thermodynamic Computed
        "Vmag[m/s]", "U[m/s]", "V[m/s]", "W[m/s]",
        "Ptot[Pa]", "Ps[Pa]", "Roh[kg/m^3]",
        "Ttot[C]", "Ts[C]", "M[]",

        # Possibly Altitude and Misc
        "Alt[m]", "AltAbs[m]",
        "Err[]", "N[]", "Toff[ms]", "Flags[]"
    ]

    assert len(HEADERS) == 36, f"[ERROR] Expected 36 headers, got {len(HEADERS)}"

    data_rows = []
    total_received = 0

    with serial.Serial(port, BAUDRATE, timeout=TIMEOUT) as ser:
        print(f"[INFO] Connected to {port}. Starting data stream...")

        # Start data transmission
        ser.write(b'TX=1\r\n')
        time.sleep(1)

        try:
            while len(data_rows) < NUM_SAMPLES:
                line = ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue

                fields = line.split('\t')
                total_received += 1

                if len(fields) != len(HEADERS):
                    print(f"[WARNING] Unexpected line length ({len(fields)}): {line}")
                    continue

                data_rows.append(fields)

        except KeyboardInterrupt:
            print("[INFO] Manual stop (Ctrl+C)")

        finally:
            ser.write(b'TX=0\r\n')
            print(f"[INFO] Finished. Collected {len(data_rows)} valid rows out of {total_received} total lines.")

    df = pd.DataFrame(data_rows, columns=HEADERS)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

    # # ----- SAVE TO CSV -----
    # df.to_csv("iprobe_data.csv", index=False)
    # print("[INFO] Data saved to 'iprobe_data.csv'")

def read_probe_data(serial_port, shared_buffer, stop_event):
    with serial.Serial(serial_port, 115200, timeout=5) as ser:
        ser.write(b'TX=1\r\n')
        time.sleep(1)

        while not stop_event.is_set():
            try:
                line = ser.readline().decode(errors='ignore').strip()
                fields = line.split('\t')
                if len(fields) == 36:
                    shared_buffer["latest"] = fields
                    shared_buffer["timestamp"] = time.time()
            except Exception:
                continue

        ser.write(b'TX=0\r\n')


def calibrate_probe(shared_data, num_samples=200, poll_interval=0.02):
    """
    Collects a number of samples from the probe to determine zero-offsets.

    Args:
        shared_data (dict): The shared dictionary for the probe thread.
        num_samples (int): The number of samples to average for calibration.
        poll_interval (float): The time to wait between samples.

    Returns:
        list: A list of 14 pressure offsets (P1 through P14).
    """
    print(f"[INFO] Collecting {num_samples} samples for probe calibration...")

    offsets_sum = np.zeros(14)
    samples_collected = 0

    # Wait for the first data point to arrive
    while shared_data.get("latest") is None:
        time.sleep(0.1)
        print("[INFO] Waiting for first probe reading...")

    with tqdm(total=num_samples, unit="sample", desc="Calibrating") as pbar:
        while samples_collected < num_samples:
            probe_fields = shared_data.get("latest")
            if probe_fields:
                # --- FIX: Convert probe data to floats before adding ---
                try:
                    # Convert the first 14 values (P1-P14) to a float array
                    numeric_pressures = np.array(probe_fields[:14], dtype=float)
                    offsets_sum += numeric_pressures
                    samples_collected += 1
                    pbar.update(1)
                except (ValueError, TypeError):
                    # This handles cases where the data might be corrupt or non-numeric
                    print("[WARNING] Skipping non-numeric probe data during calibration.")
                    pass

            time.sleep(poll_interval)

    # Calculate the average to get the final offsets
    if samples_collected > 0:
        pressure_offsets = (offsets_sum / samples_collected).tolist()
    else:
        print("[ERROR] No valid samples collected during calibration. Returning zero offsets.")
        pressure_offsets = [0.0] * 14

    return pressure_offsets


def perform_sweep_with_probe_sync(
    traverse_port="COM5",
    probe_port="COM6",
    poll_frequency=50,
    num_steps_x=10, num_steps_y=5, num_steps_z=5,
    x_bounds=(0, 300), y_bounds=(0, 295), z_bounds=(0, 203.2),
    speed_x=20, speed_y=20, speed_z=10
):
    HEADERS = [
        "time[s]", "elapsed[s]",
        "x_mm[mm]", "y_mm[mm]", "z_mm[mm]",

        # 14-hole probe differential pressures
        "P1[Pa]", "P2[Pa]", "P3[Pa]", "P4[Pa]", "P5[Pa]", "P6[Pa]", "P7[Pa]",
        "P8[Pa]", "P9[Pa]", "P10[Pa]", "P11[Pa]", "P12[Pa]", "P13[Pa]", "P14[Pa]",

        # Reference pressure and tip temperature
        "Pabs[Pa]", "Ttc[C]",

        # Attitude
        "Theta[Deg]", "Phi[Deg]", "Alpha[Deg]", "Beta[Deg]",

        # Flow velocity components
        "Vmag[m/s]", "U[m/s]", "V[m/s]", "W[m/s]",

        # Flow thermodynamics
        "Ptot[Pa]", "Ps[Pa]", "Rho[kg/m^3]", "Ttot[C]", "Ts[C]", "Mach[]",

        # System diagnostic
        "Alt[m]", "AltAbs[m]", "Err[]", "N[]", "Toff[ms]", "Flags[]"
    ]

    file_details = (
            f'_x_{abs(x_bounds[1]-x_bounds[0])}mm{speed_x}mmps{num_steps_x}points' +
            f'_y_{abs(y_bounds[1]-y_bounds[0])}mm{speed_y}mmps{num_steps_y}points' +
            f'_z_{abs(z_bounds[1]-z_bounds[0])}mm{speed_z}mmps{num_steps_z}points'
    )

    get_probe_data(probe_port, NUM_SAMPLES=10)
    print('[INFO]✅ Probe data sampling check complete!')

    x_range = np.linspace(x_bounds[0], x_bounds[1], num_steps_x)
    y_range = np.linspace(y_bounds[1], y_bounds[0], num_steps_y)
    z_range = np.linspace(z_bounds[1], z_bounds[0], num_steps_z)
    poll_interval = 1.0 / poll_frequency

    # --- TQDM SETUP ---
    # Total points for the progress bar
    total_points = len(x_range) * len(y_range) * len(z_range)

    log = []

    # Shared probe data
    shared_data = {"latest": None, "timestamp": None}
    stop_event = threading.Event()
    probe_thread = threading.Thread(target=read_probe_data, args=(probe_port, shared_data, stop_event))
    probe_thread.start()

    with Connection.open_serial_port(traverse_port) as connection:
        devices = connection.detect_devices()
        x_axis = devices[0].get_axis(1)
        y_axis = devices[1].get_axis(1)
        z_axis = devices[2].get_axis(1)

        for axis, name in zip([x_axis, y_axis, z_axis], ["X", "Y", "Z"]):
            if not axis.is_homed():
                print(f"[INFO] Homing {name} axis...")
                axis.home()

        # --- CALIBRATION STEP ---
        print("\n--- Probe Calibration ---")
        input("❗ Ensure NO FLOW condition, then press Enter to begin calibration...")

        pressure_offsets = calibrate_probe(shared_data, poll_interval=poll_interval)
        print(f"[INFO]✅ Calibration complete. Offsets determined.")

        input("\n✅ Calibration finished. Prepare for sweep, then press Enter to start...")
        print("[INFO] Starting sweep...")
        # --- END CALIBRATION STEP ---

        x_axis.settings.set("maxspeed", speed_x, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        y_axis.settings.set("maxspeed", speed_y, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        z_axis.settings.set("maxspeed", speed_z, Units.VELOCITY_MILLIMETRES_PER_SECOND)

        x_axis.move_absolute(x_range[0], Units.LENGTH_MILLIMETRES)
        y_axis.move_absolute(y_range[0], Units.LENGTH_MILLIMETRES)
        z_axis.move_absolute(z_range[0], Units.LENGTH_MILLIMETRES)

        start_time = time.time()

        # --- TQDM PROGRESS BAR ---
        # Wrap the entire sweep logic in a tqdm context manager
        with tqdm(total=total_points, unit="point", desc="3D Sweep Progress") as pbar:
            for y in y_range:
                forward = True
                for z in z_range:
                    x_path = x_range if forward else reversed(x_range)
                    for x in x_path:
                        # --- LOGIC RESTORED: Use threads to move motors simultaneously ---
                        threads = []
                        # Check current position to avoid unnecessary move commands
                        if abs(x_axis.get_position(Units.LENGTH_MILLIMETRES) - x) > 1e-3:
                            threads.append(
                                threading.Thread(target=lambda: x_axis.move_absolute(x, Units.LENGTH_MILLIMETRES)))
                        if abs(y_axis.get_position(Units.LENGTH_MILLIMETRES) - y) > 1e-3:
                            threads.append(
                                threading.Thread(target=lambda: y_axis.move_absolute(y, Units.LENGTH_MILLIMETRES)))
                        if abs(z_axis.get_position(Units.LENGTH_MILLIMETRES) - z) > 1e-3:
                            threads.append(
                                threading.Thread(target=lambda: z_axis.move_absolute(z, Units.LENGTH_MILLIMETRES)))

                        for t in threads:
                            t.start()

                        # --- LOGIC RESTORED: Poll for data WHILE the motors are moving ---
                        while any(t.is_alive() for t in threads):
                            time.sleep(poll_interval)
                            pos_x = x_axis.get_position(Units.LENGTH_MILLIMETRES)
                            pos_y = y_axis.get_position(Units.LENGTH_MILLIMETRES)
                            pos_z = z_axis.get_position(Units.LENGTH_MILLIMETRES)

                            probe_fields_raw = shared_data.get("latest")
                            if probe_fields_raw:
                                try:
                                    # --- FIX 1: Convert probe data to floats before subtracting ---
                                    pressures_raw = np.array(probe_fields_raw[:14], dtype=float)
                                    pressures_corrected = (pressures_raw - pressure_offsets).tolist()
                                    other_fields = probe_fields_raw[14:]
                                    probe_fields_final = pressures_corrected + other_fields

                                    combined_row = [
                                                       time.time(), time.time() - start_time,
                                                       pos_x, pos_y, pos_z
                                                   ] + probe_fields_final
                                    log.append(dict(zip(HEADERS, combined_row)))
                                except (ValueError, TypeError):
                                    print("[WARNING] Skipping non-numeric probe data during sweep.")
                                    pass

                        # Wait for all move threads to complete
                        for t in threads:
                            t.join()

                        # --- PROGRESS BAR UPDATE ---
                        # Update the bar only AFTER the target point is reached and all
                        # in-transit data for that segment has been logged.
                        pbar.update(1)

                    forward = not forward

    stop_event.set()
    probe_thread.join()

    df = pd.DataFrame(log, columns=HEADERS)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass  # Leave non-numeric columns (like hex strings) unchanged
    print(f"[INFO]✅ Done. Collected {len(df)} rows.")
    return df, file_details


def export_probe_data_to_vtk_point_data(df, filename):
    """
    Convert 14-hole probe + traverse data to a VTK PolyData file (.vtp).
    This file will contain point data.

    Parameters:
        df (pd.DataFrame): Probe data with specified headers
        filename (str): Output file name (should end in .vtp or .vtk)
    """

    # Convert mm to meters for x, y, z
    coords = df[["x_mm[mm]", "y_mm[mm]", "z_mm[mm]"]].to_numpy() / 1000.0
    cloud = pv.PolyData(coords)

    # Add velocity vector
    if all(col in df.columns for col in ["U[m/s]", "V[m/s]", "W[m/s]"]):
        velocity = df[["U[m/s]", "V[m/s]", "W[m/s]"]].to_numpy()
        cloud["Velocity"] = velocity

    # Add scalar fields (optional but useful for ParaView)
    scalar_fields = [
        "Vmag[m/s]", "Ptot[Pa]", "Ps[Pa]", "Pabs[Pa]",
        "Ttot[C]", "Ts[C]", "Ttc[C]", "Rho[kg/m^3]",
        "Alpha[Deg]", "Beta[Deg]", "Theta[Deg]", "Phi[Deg]",
        "Mach[]"
    ]

    for field in scalar_fields:
        if field in df.columns:
            cloud[field] = df[field].to_numpy()

    # Save the file
    cloud.save(filename)
    print(f"[INFO] Generated point data VTK file: {filename}")


def export_interpolated_data_to_vtk_cell_data(df, filename, grid_spacing_mm=20.0):
    """
    Interpolate 14-hole probe + traverse data onto a structured grid
    and export it as a VTK StructuredGrid file (.vts).
    This file will contain cell data. It dynamically adjusts for 2D or 3D sweeps.

    Parameters:
        df (pd.DataFrame): Probe data with specified headers
        filename (str): Output file name (should end in .vts)
        grid_spacing_mm (float): Desired spacing for the interpolation grid in millimeters.
    """

    print(f"Interpolating data with a grid spacing of {grid_spacing_mm} mm...")

    # Convert mm to meters for x, y, z for consistency
    coords_mm = df[["x_mm[mm]", "y_mm[mm]", "z_mm[mm]"]].to_numpy()
    points_meters = coords_mm / 1000.0

    # Determine bounds of the data in meters
    x_min, x_max = points_meters[:, 0].min(), points_meters[:, 0].max()
    y_min, y_max = points_meters[:, 1].min(), points_meters[:, 1].max()
    z_min, z_max = points_meters[:, 2].min(), points_meters[:, 2].max()

    # Use a small tolerance to check for flat dimensions
    tolerance_m = 1e-6  # 1 micrometer

    is_x_flat = (x_max - x_min) < tolerance_m
    is_y_flat = (y_max - y_min) < tolerance_m
    is_z_flat = (z_max - z_min) < tolerance_m

    print(f"  X-dimension range: {x_max - x_min:.6e} m (Flat: {is_x_flat})")
    print(f"  Y-dimension range: {y_max - y_min:.6e} m (Flat: {is_y_flat})")
    print(f"  Z-dimension range: {z_max - z_min:.6e} m (Flat: {is_z_flat})")

    grid_spacing_m = grid_spacing_mm / 1000.0

    # Handle case where all dimensions are flat (single point)
    if is_x_flat and is_y_flat and is_z_flat:
        print("  Warning: All dimensions are flat. Cannot perform interpolation.")
        # Create a single point structured grid
        structured_grid = pv.StructuredGrid(np.array([x_min]), np.array([y_min]), np.array([z_min]))

        structured_grid.save(filename)
        print(f"Saved single-point structured grid to {filename}")
        return

    # griddata interpolation (only include varying dimensions)
    interp_points_coords = []
    interp_grid_slices = []

    # pyvista StructuredGrid (always 3D)
    full_grid_x_vals = np.array([x_min]) if is_x_flat else np.linspace(x_min, x_max, int(np.ceil(
        (x_max - x_min) / grid_spacing_m)) + 1)
    full_grid_y_vals = np.array([y_min]) if is_y_flat else np.linspace(y_min, y_max, int(np.ceil(
        (y_max - y_min) / grid_spacing_m)) + 1)
    full_grid_z_vals = np.array([z_min]) if is_z_flat else np.linspace(z_min, z_max, int(np.ceil(
        (z_max - z_min) / grid_spacing_m)) + 1)

    # Determine the varying dimensions for griddata
    if not is_x_flat:
        interp_points_coords.append(points_meters[:, 0])
        interp_grid_slices.append(
            slice(full_grid_x_vals.min(), full_grid_x_vals.max(), complex(0, len(full_grid_x_vals))))
    if not is_y_flat:
        interp_points_coords.append(points_meters[:, 1])
        interp_grid_slices.append(
            slice(full_grid_y_vals.min(), full_grid_y_vals.max(), complex(0, len(full_grid_y_vals))))
    if not is_z_flat:
        interp_points_coords.append(points_meters[:, 2])
        interp_grid_slices.append(
            slice(full_grid_z_vals.min(), full_grid_z_vals.max(), complex(0, len(full_grid_z_vals))))

    if not interp_points_coords:
        raise ValueError("No varying dimensions detected for interpolation.")

    points_for_griddata = np.column_stack(interp_points_coords)

    # Create the target grid for griddata
    target_grid_coords_for_interp = np.mgrid[tuple(interp_grid_slices)]
    interp_result_shape = target_grid_coords_for_interp[0].shape

    # Create the full 3D mesh grid for pyvista StructuredGrid
    grid_x, grid_y, grid_z = np.meshgrid(
        full_grid_x_vals,
        full_grid_y_vals,
        full_grid_z_vals,
        indexing='ij'
    )
    structured_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)

    # Scalar fields to interpolate
    scalar_fields = [
        "Vmag[m/s]", "Ptot[Pa]", "Ps[Pa]", "Pabs[Pa]",
        "Ttot[C]", "Ts[C]", "Ttc[C]", "Rho[kg/m^3]",
        "Alpha[Deg]", "Beta[Deg]", "Theta[Deg]", "Phi[Deg]",
        "Mach[]"
    ]

    # Build indexing tuple for placing interpolated data into the full 3D array
    full_3d_slice_indices = []
    if not is_x_flat:
        full_3d_slice_indices.append(slice(None))
    else:
        full_3d_slice_indices.append(0)
    if not is_y_flat:
        full_3d_slice_indices.append(slice(None))
    else:
        full_3d_slice_indices.append(0)
    if not is_z_flat:
        full_3d_slice_indices.append(slice(None))
    else:
        full_3d_slice_indices.append(0)
    full_3d_slice_tuple = tuple(full_3d_slice_indices)

    # Interpolate scalar fields
    for field in scalar_fields:
        if field in df.columns:
            print(f"  Interpolating scalar field: {field}")
            values = df[field].values

            interpolated_values_raw = griddata(
                points_for_griddata,
                values,
                tuple(target_grid_coords_for_interp),
                method='linear',
                fill_value=np.nan
            ).reshape(interp_result_shape)

            final_interpolated_values = np.full(grid_x.shape, np.nan)
            final_interpolated_values[full_3d_slice_tuple] = interpolated_values_raw.squeeze()

            # CHANGED: Assign data to points using Fortran ('F') ordering to prevent distortion.
            # Using .point_data accessor for clarity.
            structured_grid.point_data[field] = final_interpolated_values.flatten(order='F')
        else:
            print(f"  Warning: Field '{field}' not found in DataFrame. Skipping interpolation.")

    # Interpolate velocity vector if available
    if all(col in df.columns for col in ["U[m/s]", "V[m/s]", "W[m/s]"]):
        print("  Interpolating velocity vector (U, V, W)...")
        velocity_components = df[["U[m/s]", "V[m/s]", "W[m/s]"]].values

        interpolated_u_raw = griddata(points_for_griddata, velocity_components[:, 0],
                                      tuple(target_grid_coords_for_interp), method='linear', fill_value=np.nan).reshape(
            interp_result_shape)
        interpolated_v_raw = griddata(points_for_griddata, velocity_components[:, 1],
                                      tuple(target_grid_coords_for_interp), method='linear', fill_value=np.nan).reshape(
            interp_result_shape)
        interpolated_w_raw = griddata(points_for_griddata, velocity_components[:, 2],
                                      tuple(target_grid_coords_for_interp), method='linear', fill_value=np.nan).reshape(
            interp_result_shape)

        final_interpolated_u = np.full(grid_x.shape, np.nan)
        final_interpolated_v = np.full(grid_x.shape, np.nan)
        final_interpolated_w = np.full(grid_x.shape, np.nan)

        final_interpolated_u[full_3d_slice_tuple] = interpolated_u_raw.squeeze()
        final_interpolated_v[full_3d_slice_tuple] = interpolated_v_raw.squeeze()
        final_interpolated_w[full_3d_slice_tuple] = interpolated_w_raw.squeeze()

        # CHANGED: Flatten each component with Fortran ('F') ordering and then stack them.
        # This ensures the vectors are correctly associated with their grid points.
        flat_u = final_interpolated_u.flatten(order='F')
        flat_v = final_interpolated_v.flatten(order='F')
        flat_w = final_interpolated_w.flatten(order='F')

        # Using .point_data accessor for clarity.
        structured_grid.point_data["Velocity"] = np.stack((flat_u, flat_v, flat_w), axis=1)
    else:
        print("  Warning: Velocity components (U, V, W) not found in DataFrame. Skipping.")

    # CHANGED: Convert the point data to cell data before saving.
    # This creates a new grid where data is associated with cells, not points.
    print("Converting point data to cell data...")
    final_grid_with_cell_data = structured_grid.point_data_to_cell_data()

    # Save the final grid containing cell data
    final_grid_with_cell_data.save(filename)
    print(f"Successfully saved interpolated cell data to {filename}")

def export_interpolated_data_to_vtk_cell_data_pv_interpolate(df, filename, grid_spacing_mm=5.0):
    """
    Interpolate 14-hole probe + traverse data onto a structured grid
    using PyVista's interpolation filter and export it as a VTK StructuredGrid file (.vts)
    with cell data.
    """
    print(f"Interpolating data with PyVista filter with a grid spacing of {grid_spacing_mm} mm...")

    # 1. Create a PyVista PolyData object from the raw DataFrame (Source Data)
    coords_mm = df[["x_mm[mm]", "y_mm[mm]", "z_mm[mm]"]].to_numpy()
    source_points_meters = coords_mm / 1000.0 # Convert to meters

    # Create the PolyData object for your scattered probe points
    source_mesh = pv.PolyData(source_points_meters)

    # Add scalar and vector data to the source_mesh's point_data
    scalar_fields = [
        "Vmag[m/s]", "Ptot[Pa]", "Ps[Pa]", "Pabs[Pa]",
        "Ttot[C]", "Ts[C]", "Ttc[C]", "Rho[kg/m^3]",
        "Alpha[Deg]", "Beta[Deg]", "Theta[Deg]", "Phi[Deg]",
        "Mach[]"
    ]
    for field in scalar_fields:
        if field in df.columns:
            source_mesh.point_data[field] = df[field].values
        else:
            print(f"  Warning: Field '{field}' not found in DataFrame for source_mesh. Skipping.")

    if all(col in df.columns for col in ["U[m/s]", "V[m/s]", "W[m/s]"]):
        source_mesh.point_data["Velocity"] = df[["U[m/s]", "V[m/s]", "W[m/s]"]].values
    else:
        print("  Warning: Velocity components (U, V, W) not found for source_mesh. Skipping.")


    # 2. Determine bounds and create the target StructuredGrid
    x_min, x_max = source_points_meters[:, 0].min(), source_points_meters[:, 0].max()
    y_min, y_max = source_points_meters[:, 1].min(), source_points_meters[:, 1].max()
    z_min, z_max = source_points_meters[:, 2].min(), source_points_meters[:, 2].max()

    grid_spacing_m = grid_spacing_mm / 1000.0

    # These are always 3D for PyVista's StructuredGrid,
    # handling flat dimensions by having a single value
    is_x_flat = (x_max - x_min) < 1e-6
    is_y_flat = (y_max - y_min) < 1e-6
    is_z_flat = (z_max - z_min) < 1e-6

    full_grid_x_vals = np.array([x_min]) if is_x_flat else np.linspace(x_min, x_max, int(np.ceil(
        (x_max - x_min) / grid_spacing_m)) + 1)
    full_grid_y_vals = np.array([y_min]) if is_y_flat else np.linspace(y_min, y_max, int(np.ceil(
        (y_max - y_min) / grid_spacing_m)) + 1)
    full_grid_z_vals = np.array([z_min]) if is_z_flat else np.linspace(z_min, z_max, int(np.ceil(
        (z_max - z_min) / grid_spacing_m)) + 1)

    grid_x, grid_y, grid_z = np.meshgrid(
        full_grid_x_vals,
        full_grid_y_vals,
        full_grid_z_vals,
        indexing='ij'
    )
    target_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)

    # 3. Perform the interpolation using PyVista's filter
    print("  Performing interpolation using PyVista filter...")
    # The interpolate method interpolates source_mesh's point data onto target_grid's points
    # It will automatically handle all point_data arrays from source_mesh.
    # The 'null_value' is what values outside the interpolation domain will be filled with.
    # The 'radius' or 'kernel' arguments can control interpolation method if needed.
    interpolated_grid = target_grid.interpolate(source_mesh, null_value=np.nan, progress_bar=True)


    # 4. Convert point data to cell data (still needed as interpolate produces point data by default)
    print("Converting point data to cell data...")
    final_grid_with_cell_data = interpolated_grid.point_data_to_cell_data()

    # 5. Save the final grid
    final_grid_with_cell_data.save(filename)
    print(f"Successfully saved interpolated cell data to {filename}")
