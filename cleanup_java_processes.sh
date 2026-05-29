#!/bin/bash
# Script to clean up Java/Spark processes from NXCALS operations
# This helps resolve hanging issues in parallel processing

echo "=== Checking for Java/Spark processes ==="
# Exclude this script and grep processes from the search
JAVA_PROCESSES=$(ps aux | grep java | grep -v grep | grep -v cleanup_java_processes)

if [ -n "$JAVA_PROCESSES" ]; then
    echo "$JAVA_PROCESSES"
    echo ""
    echo "=== Found Java processes. Cleaning up... ==="

    echo "Attempting graceful shutdown of Spark processes..."
    pkill -f SparkSubmit

    sleep 2

    echo "Checking if processes are still running..."
    RUNNING=$(ps aux | grep -E "(SparkSubmit|pyspark)" | grep -v grep | grep -v cleanup_java_processes | wc -l)

    if [ $RUNNING -gt 0 ]; then
        echo "Some processes still running. Force killing..."
        pkill -9 -f SparkSubmit
        sleep 1
    fi

    echo "Final check for any remaining processes..."
    REMAINING=$(ps aux | grep -E "(spark|nxcals)" | grep -v grep | grep -v cleanup_java_processes)

    if [ -n "$REMAINING" ]; then
        echo "$REMAINING"
        echo "WARNING: Some processes may still be running."
        echo "You can manually kill with: kill -9 <PID>"
    else
        echo "✅ All Java/Spark processes cleaned up successfully!"
    fi
else
    echo "✅ No Java processes found - system is clean!"
fi

echo ""
echo "=== Process cleanup complete ==="