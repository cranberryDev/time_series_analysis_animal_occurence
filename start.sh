#!/bin/sh
# entrypoint for railway or other platform with PORT variable
# start the gunicorn server
exec gunicorn -w 4 -b 0.0.0.0:${PORT:-8000} ui:app
