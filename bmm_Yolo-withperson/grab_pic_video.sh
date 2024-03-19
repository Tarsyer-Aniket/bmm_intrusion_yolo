#!/bin/bash

Date=$(date +'%d-%m-%Y')

ffmpeg -ss 00:00:03 -i "/tmp/awarpur_safety.mkv" -frames:v 1 -q:v 2 ${Date}_amarpur.jpg
