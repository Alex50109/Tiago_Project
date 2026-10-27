find . -type d -wholename 'depth_server|.git' \
| xargs tar cf - \
| ssh pal@$1 tar xf - -C /home/pal/catkin_ws/src/Tiago_Project
