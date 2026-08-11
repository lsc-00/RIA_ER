카메라 설정열기

ffplay -f v4l2 \
  -input_format mjpeg \
  -video_size 1280x720 \
  -framerate 30 \
  /dev/video8
  
  
  점유 중인 프로세스 확인
  sudo lsof /dev/video8
