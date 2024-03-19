#!/bin/bash

Date=$(date +'%d-%m-%Y')

ffmpeg -rtsp_transport tcp -ss 00:00:03 -i "rtsp://admin:Tarsyer123@192.168.1.$1:554/Streaming/channels/101" -frames:v 1 -q:v 2 ${Date}_camera_no.jpg