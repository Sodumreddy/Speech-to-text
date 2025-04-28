#!/bin/bash

# Check if MongoDB is running
if ! pgrep -x "mongod" > /dev/null; then
    echo "MongoDB is not running. Starting MongoDB..."
    mongod --fork --logpath /tmp/mongodb.log
    sleep 5
fi

# Start the application
echo "Starting the application..."
python run.py &

# Wait for the application to start
sleep 5

# Start ngrok
echo "Starting ngrok..."
python start_ngrok.py

# Keep the script running
echo "Application and ngrok are running. Press Ctrl+C to stop."
wait 