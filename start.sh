#!/bin/bash

echo "Building vector database..."
python call_test.py

echo "Starting backend..."
python backend.py