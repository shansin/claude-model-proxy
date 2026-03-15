#!/usr/bin/env bash

# benchmark_model.sh - Benchmark all available Ollama models sequentially

# Determine script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ENV_FILE="$DIR/.env"

# Extract configuration from .env if present
if [ -f "$ENV_FILE" ]; then
    OLLAMA_PROXY_CTX=$(grep -E '^OLLAMA_CONTEXT_SIZE_DEFAULT=' "$ENV_FILE" | cut -d '=' -f2 | tr -d ' "')
    OLLAMA_BASE_URL_ENV=$(grep -E '^OLLAMA_BASE_URL=' "$ENV_FILE" | cut -d '=' -f2 | tr -d ' "')
fi

# Fallback values
CONTEXT_SIZE=${OLLAMA_PROXY_CTX:-128000}
OLLAMA_URL=${OLLAMA_BASE_URL_ENV:-http://localhost:11434}

echo "=================================================="
echo "LLM Benchmarker"
echo "=================================================="
echo "Using Context Size: $CONTEXT_SIZE"
echo "Using Ollama URL  : $OLLAMA_URL"
echo ""

# Prompts
PROMPT_1="Create a simple HTML page with a styled button and some text. Provide only the code."
PROMPT_2="Write a python script that implements a simple web server using built-in http.server module. Provide only the code."

# Check if Ollama is running
if ! curl -s "${OLLAMA_URL}/api/version" > /dev/null; then
    echo "Error: Cannot connect to Ollama at $OLLAMA_URL. Is the server running?"
    exit 1
fi

# Get available models
MODELS=$(ollama list | tail -n +2 | awk '{print $1}')

if [ -z "$MODELS" ]; then
    echo "No models found via 'ollama list'. Please download some models first."
    exit 1
fi

# Initialize report file
REPORT_FILE="$DIR/benchmark-reports/$(date +"%Y%m%d_%H%M")_report.csv"
echo "Model,Size,Quant,Prompt,Prompt Tokens/sec,Gen Tokens/sec,First Response (s),Total Duration (s),Load Duration (s),Tokens Generated" > "$REPORT_FILE"

echo "Starting Benchmark..."

for MODEL in $MODELS; do
    echo "--------------------------------------------------"
    
    # Check model capabilities and extract model details
    SHOW_RESPONSE=$(curl -s "${OLLAMA_URL}/api/show" -d "{\"model\": \"$MODEL\"}")
    
    if [[ "$SHOW_RESPONSE" != *"\"completion\""* ]]; then
        echo "Skipping non-generative (e.g., embedding) model: $MODEL"
        continue
    fi

    MODEL_DETAILS=$(python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    details = d.get('details', {})
    print(f\"{details.get('parameter_size', 'Unknown')} {details.get('quantization_level', 'Unknown')}\")
except Exception:
    print('Unknown Unknown')
" <<< "$SHOW_RESPONSE")
    
    PARAM_SIZE=$(echo "$MODEL_DETAILS" | awk '{print $1}')
    QUANT_LEVEL=$(echo "$MODEL_DETAILS" | awk '{print $2}')

    echo "Target Model: $MODEL ($PARAM_SIZE | $QUANT_LEVEL)"
    
    # Explicitly load the model into memory
    echo "  -> Loading model into memory (can take some time)..."
    curl -s "${OLLAMA_URL}/api/generate" -d "{\"model\": \"$MODEL\"}" > /dev/null
    
    for i in 1 2; do
        if [ "$i" -eq 1 ]; then
            PROMPT="$PROMPT_1"
            PROMPT_NAME="HTML_Generation"
        else
            PROMPT="$PROMPT_2"
            PROMPT_NAME="Python_Script"
        fi

        echo "  -> Running prompt: $PROMPT_NAME"

        # Query Ollama API natively with timing outputs
        RESPONSE=$(curl -s -X POST "${OLLAMA_URL}/api/generate" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"prompt\": \"$PROMPT\",
                \"stream\": false,
                \"options\": {
                    \"num_ctx\": $CONTEXT_SIZE
                }
            }")

        # Parse response safely using Python mapping
        METRICS=$(python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(f\"{d.get('eval_count', 0)} {d.get('eval_duration', 0)} {d.get('prompt_eval_count', 0)} {d.get('prompt_eval_duration', 0)} {d.get('load_duration', 0)}\")
except Exception:
    print('0 0 0 0 0')
" <<< "$RESPONSE")
        
        EVAL_COUNT=$(echo "$METRICS" | awk '{print $1}')
        EVAL_DURATION_NS=$(echo "$METRICS" | awk '{print $2}')
        PROMPT_EVAL_COUNT=$(echo "$METRICS" | awk '{print $3}')
        PROMPT_EVAL_DURATION_NS=$(echo "$METRICS" | awk '{print $4}')
        LOAD_DURATION_NS=$(echo "$METRICS" | awk '{print $5}')

        # Handle failed requests safely
        if [ "$EVAL_COUNT" -eq 0 ]; then
            echo "     Failed to get valid completion."
            echo "$MODEL,$PARAM_SIZE,$QUANT_LEVEL,$PROMPT_NAME,Error,Error,Error,Error,Error,Error" >> "$REPORT_FILE"
            continue
        fi

        # Metrics computations (1 second = 1,000,000,000 ns)
        TPS=$(awk "BEGIN {printf \"%.2f\", $EVAL_COUNT / ($EVAL_DURATION_NS / 1000000000)}")
        
        # Calculate prompt processing TPS
        PROMPT_TPS="0.00"
        if [ "$PROMPT_EVAL_DURATION_NS" -gt 0 ] && [ "$PROMPT_EVAL_COUNT" -gt 0 ]; then
            PROMPT_TPS=$(awk "BEGIN {printf \"%.2f\", $PROMPT_EVAL_COUNT / ($PROMPT_EVAL_DURATION_NS / 1000000000)}")
        fi
        
        LOAD_DUR=$(awk "BEGIN {printf \"%.2f\", $LOAD_DURATION_NS / 1000000000}")
        FIRST_RES=$(awk "BEGIN {printf \"%.2f\", ($LOAD_DURATION_NS + $PROMPT_EVAL_DURATION_NS) / 1000000000}")
        TOTAL_DUR=$(awk "BEGIN {printf \"%.2f\", ($LOAD_DURATION_NS + $PROMPT_EVAL_DURATION_NS + $EVAL_DURATION_NS) / 1000000000}")

        echo "     Size / Quant     : $PARAM_SIZE / $QUANT_LEVEL"
        echo "     Prompt Tokens/sec: $PROMPT_TPS ($PROMPT_EVAL_COUNT tokens)"
        echo "     Gen Tokens/sec   : $TPS"
        echo "     Load Duration    : $LOAD_DUR s"
        echo "     First Response   : $FIRST_RES s"
        echo "     Total Duration   : $TOTAL_DUR s"
        echo "     Generated        : $EVAL_COUNT tokens"
        
        echo "$MODEL,$PARAM_SIZE,$QUANT_LEVEL,$PROMPT_NAME,$PROMPT_TPS,$TPS,$FIRST_RES,$TOTAL_DUR,$LOAD_DUR,$EVAL_COUNT" >> "$REPORT_FILE"
    done
    
    # Unload the model to free VRAM for the next benchmark
    echo "  -> Unloading $MODEL..."
    curl -s "${OLLAMA_URL}/api/generate" -d "{\"model\": \"$MODEL\", \"keep_alive\": 0}" > /dev/null
done

echo "=================================================="
echo "Benchmark completed successfully."
echo "Final report saved to: $REPORT_FILE"
echo ""
echo "Summary:"
column -t -s ',' "$REPORT_FILE"
