import numpy as np
views = 1
y_rot_range_list = [0,15, 30, 60, 90, 120, 150, 180]
sample_num = 10
if views == 1:
    stepsize = 360 
else:
    stepsize = 360 / views

print(f"stepsize: {stepsize} deg")

##No hard mode rendering before range(2)


for y_rot_range in y_rot_range_list:
    y_rot_list = []
    for _ in range(sample_num):
        for j in range(1):
            current_rot_value = 0
            metastring = ""
            for i in range(views):
                current_rot_value += stepsize
                counter = 0
                while True:
                    counter+=1
                    angle_rand = np.random.rand(3)
                    y_rot = current_rot_value + angle_rand[0] *  2 * y_rot_range - y_rot_range
                    if j == 0:
                        x_rot = 20 + angle_rand[1] * 10
                    else:
                        x_rot = angle_rand[1] * 45
                    if j== 0:
                        dist = 0.65 + angle_rand[2] * 0.35
                    #elif counter >= 5:
                    #    dist = 0.75 + angle_rand[2] * 0.35
                    #else:
                    #    dist = 0.60 + angle_rand[2] * 0.40
                    param = [y_rot, x_rot, 0, dist, 35, 32, 1.75]
                    y_rot_list.append(y_rot)
                    break
    
    print(y_rot_range, y_rot_list)