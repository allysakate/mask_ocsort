import os
import csv
from gpxcsv import gpxtolist
import haversine as hs

base_dir = "/media/catchall/OneTouch/1_PhD/Dissertation/Evaluation/tracking/Calibration"
for num in [0, 1, 2]:
    print(f"Num: {num}")
    input_file = os.path.join(base_dir, f"morning_ride_{num}.gpx")
    output_file = os.path.join(base_dir, f"morning_ride_{num}.csv")
    gpx_list = gpxtolist(input_file)
    keys = gpx_list[0].keys()
    a_file = open(output_file, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(gpx_list)
    a_file.close()
    first_loc = gpx_list[0]
    last_loc = gpx_list[-1]
    for idx, gpx in enumerate(gpx_list):
        print(idx, gpx["lat"], gpx["lon"])
    loc1 = (first_loc["lat"], first_loc["lon"])  # (lat, lon) 14.419275, 121.012529
    loc2 = (last_loc["lat"], last_loc["lon"])  # (lat, lon) 14.419629, 121.012552
    print(num, hs.haversine(loc1, loc2, unit=hs.Unit.INCHES))

loc1 = (14.419629, 121.012552)  # 14.419629,121.012552
loc2 = (14.419511, 121.012585)  # 14.419511,121.012585
print(hs.haversine(loc1, loc2, unit=hs.Unit.METERS))  # 535.1883361501633
