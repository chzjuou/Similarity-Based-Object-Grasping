#!/bin/sh

gnome-terminal --tab -- /bin/bash -c "python capture_background.py" 

gnome-terminal --tab -- /bin/bash -c "sh detic.sh" 

gnome-terminal --tab -- /bin/bash -c "cd wrs/robot_sim/robots/ur5e;

			python ur5e_grasping.py"
