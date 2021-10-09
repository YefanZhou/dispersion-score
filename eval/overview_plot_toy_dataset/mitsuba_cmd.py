import os
from os.path import join
import glob

# pts_xml_folder = "plot/pts_vis"
# pts_xml_paths = glob.glob(join(pts_xml_folder, '*.xml'))
# pts_xml_paths.sort()

# for pts_xml_path in pts_xml_paths:
#     cmd = f'/Applications/Mitsuba.app/Contents/MacOS/mitsuba {pts_xml_path}'
#     os.system(cmd)



pts_xml_folder = "../../dataset/toy_dataset/vis/augment_demo_submission/points"
pts_xml_paths = glob.glob(join(pts_xml_folder, '*.xml'))
pts_xml_paths.sort()
for pts_xml_path in pts_xml_paths:
    cmd = f'/Applications/Mitsuba.app/Contents/MacOS/mitsuba {pts_xml_path}'
    os.system(cmd)