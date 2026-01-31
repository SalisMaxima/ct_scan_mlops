#!/bin/bash
# Test all Dockerfiles to ensure they build successfully
# Usage: bash scripts/test_dockerfiles.sh

set -e  # Exit on error

echo "=========================================="
echo "Docker Build Test Suite - v2.0"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Function to test a Dockerfile
test_dockerfile() {
    local dockerfile=$1
    local tag=$2
    local description=$3

    echo -n "Testing ${description}... "

    if docker build -f "${dockerfile}" -t "${tag}" . > /tmp/docker_build_${tag//[:\/]/_}.log 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  See log: /tmp/docker_build_${tag//[:\/]/_}.log"
        ((FAILED++))
        return 1
    fi
}

# Function to test if NVIDIA Docker is available
check_nvidia_docker() {
    if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Test 1: CPU Training Dockerfile
echo "Test 1: CPU Training Dockerfile"
test_dockerfile "dockerfiles/train.dockerfile" "ct-scan-train:cpu-test" "train.dockerfile"
echo ""

# Test 2: GPU Training Dockerfile (skip if no GPU)
echo "Test 2: GPU Training Dockerfile"
if check_nvidia_docker; then
    test_dockerfile "dockerfiles/train_cuda.dockerfile" "ct-scan-train:cuda-test" "train_cuda.dockerfile"
else
    echo -e "${YELLOW}⊘ SKIPPED (no NVIDIA Docker available)${NC}"
    ((SKIPPED++))
fi
echo ""

# Test 3: API Dockerfile
echo "Test 3: API Dockerfile (standalone)"
test_dockerfile "dockerfiles/api.dockerfile" "ct-scan-api:test" "api.dockerfile"
echo ""

# Test 4: Cloud Run API Dockerfile
echo "Test 4: API Dockerfile (Cloud Run)"
test_dockerfile "dockerfiles/api.cloudrun.dockerfile" "ct-scan-api:cloudrun-test" "api.cloudrun.dockerfile"
echo ""

# Test 5: Drift API Dockerfile
echo "Test 5: Drift Monitoring API Dockerfile"
test_dockerfile "dockerfiles/drift_api.dockerfile" "ct-scan-drift:test" "drift_api.dockerfile"
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"
echo "Total:   $((PASSED + FAILED + SKIPPED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! 🎉${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Test running containers: docker run --rm <image-name>"
    echo "  2. Test volume mounts: docker run -v \$(pwd)/outputs:/app/outputs <image-name>"
    echo "  3. Test dual pathway: Ensure features extracted first"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs in /tmp/${NC}"
    echo ""
    echo "Common fixes:"
    echo "  - Ensure Docker daemon is running"
    echo "  - Check .dockerignore doesn't exclude necessary files"
    echo "  - Verify all referenced files exist"
    exit 1
fi
