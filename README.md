# RIA_ER
일합시다

싫은데요

해야만 합니다

어서오고

#vision part

#1.start realsense launch code 
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true

#2. Then, another terminal starts yolo detecting node for taking frame 
ros2 run fried_food_vision yolo_detection_node

#3. Last, put this command third terminal for getting depth  
ros2 run fried_food_vision depth_position_node
