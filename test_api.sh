#!/bin/bash

# Test CT Scan API with different classes

API_URL="https://ct-scan-api-777769481436.europe-west1.run.app"

echo "================================"
echo "Testing CT Scan Classification API"
echo "================================"
echo ""

# Test 1: Adenocarcinoma
echo "1. Testing ADENOCARCINOMA..."
curl -s -X POST "$API_URL/predict" \
  -F "file=@data/raw/chest-ctscan-images/Data/test/adenocarcinoma/000108 (3).png" \
  | python3 -m json.tool
echo ""

# Test 2: Normal
echo "2. Testing NORMAL..."
curl -s -X POST "$API_URL/predict" \
  -F "file=@data/raw/chest-ctscan-images/Data/test/normal/7 - Copy - Copy.png" \
  | python3 -m json.tool
echo ""

# Test 3: Large Cell Carcinoma
echo "3. Testing LARGE CELL CARCINOMA..."
curl -s -X POST "$API_URL/predict" \
  -F "file=@data/raw/chest-ctscan-images/Data/test/large.cell.carcinoma/000131 (2).png" \
  | python3 -m json.tool
echo ""

# Test 4: Squamous Cell Carcinoma
echo "4. Testing SQUAMOUS CELL CARCINOMA..."
curl -s -X POST "$API_URL/predict" \
  -F "file=@data/raw/chest-ctscan-images/Data/test/squamous.cell.carcinoma/000132 (4).png" \
  | python3 -m json.tool
echo ""

echo "================================"
echo "API Testing Complete!"
echo "================================"
