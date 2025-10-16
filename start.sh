#!/bin/sh

gnome-terminal --tab -- /bin/bash -c "python object_recognition.py"
			
gnome-terminal --tab -- /bin/bash -c "python grasp_execution.py"




