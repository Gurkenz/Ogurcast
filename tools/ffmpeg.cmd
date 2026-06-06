@echo off
if not defined OGURCAST_FFMPEG (
  echo Error: FFmpeg executable is not resolved. 1>&2
  exit /b 9009
)
"%OGURCAST_FFMPEG%" %*
