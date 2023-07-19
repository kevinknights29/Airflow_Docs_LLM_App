#!/bin/bash
# db_migration.sh
# This script copies the files from the specified directory to the current directory.

# To make this script executable, run the following command:
# chmod +x db_migration.sh

# Usage: ./db_migration.sh DB_FOLDER

# Ensure script has been called with an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DB_FOLDER"
    exit 1
fi

DB_FOLDER=$1

# Check if the directory exists
if [ ! -d "$DB_FOLDER" ]; then
    echo "Error: Directory $DB_FOLDER does not exist."
    exit 1
fi

# Check if the directory is readable
if [ ! -r "$DB_FOLDER" ]; then
    echo "Error: Directory $DB_FOLDER is not readable."
    exit 1
fi

# Copy the files
cp -r "$DB_FOLDER"/* .

echo "Files copied successfully."
