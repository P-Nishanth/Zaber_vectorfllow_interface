#main.py
import Universe as Uni
import pandas as pd

# Uni.rotate_device(angle=90, speed_dps=24)

# Uni.move_probe_to(x_mm=100, y_mm=100, z_mm=100, speed_xy=53, speed_z=26)
#
# df = Uni.perform_volume_sweep(
#     x_bounds=(0, 300), y_bounds=(0, 240), z_bounds=(0, 200),
#     num_steps_x=2, num_steps_y=24, num_steps_z=20,
#     speed_x=50, speed_y=50, speed_z=26
# )
# print(df)

# df = Uni.poll_positions_25hz(duration_sec=5)
# print(df.head())

# df = Uni.poll_positions_25hz()
# print(df)

# df = Uni.perform_volume_sweep_with_polling(
#     x_bounds=(0, 100), y_bounds=(100, 200), z_bounds=(50, 100),
#     num_steps_x=2, num_steps_y=5, num_steps_z=5,
#     speed_x=50, speed_y=25, speed_z=15
# )
# print(df)

# df = Uni.get_probe_data(port='COM6')
# print(df)

(df, file_details) = Uni.perform_sweep_with_probe_sync(
                    x_bounds=(0, 100), y_bounds=(100, 200), z_bounds=(50, 100),
                    num_steps_x=2, num_steps_y=5, num_steps_z=5,
                    speed_x=50, speed_y=50, speed_z=26
)
print(df)
# # # #
# # # #
# filename=f'dualProp_50percent_7x5x3_{file_details}'

# df.to_csv(filename+".csv",index=False)

# filename="fanground4__x_300mm50mmps2points_y_240mm50mmps24points_z_200mm26mmps20points"
# df = pd.read_csv(filename+'.csv')
# df["x_mm[mm]"], df["y_mm[mm]"], df["z_mm[mm]"] = df["y_mm[mm]"].copy(), df["z_mm[mm]"].copy(), df["x_mm[mm]"].copy()
# df["x_mm[mm]"], df["y_mm[mm]"] = -df["y_mm[mm]"].copy(), df["x_mm[mm]"].copy()
# # # # # print(df)
# Uni.export_probe_data_to_vtk_point_data(df, filename=filename+'.vtk')
# # # #
# Uni.export_interpolated_data_to_vtk_cell_data(df, filename+'.vts', grid_spacing_mm=10)
