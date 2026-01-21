#!/bin/bash
# start_inspection_terminals_supervised.sh
# Auto-start all inspection clients and orchestrator in separate terminals
# Automatically restarts if any script crashes
# Logs outputs in /home/nvidia/logs

LOG_DIR="/home/nvidia/logs"
mkdir -p $LOG_DIR

# Function to start a script in a terminal with restart
start_terminal() {
    local name="$1 dock"
    local cmd="$2"
    local log_file="$3"

    gnome-terminal -- bash -c "
        while true; do
            echo \"[\$(date)] Starting $name...\"
            $cmd >> \"$log_file\" 2>&1
            echo \"[\$(date)] $name crashed! Restarting in 2 seconds...\"
            sleep 2
        done
        exec bash
    "
}

# Start Body Client
start_terminal "Body Client" \
"python /home/nvidia/big400_bottle_inspection/inference_clients/body_client.py \
--daemon --cam-index 0 --api-url http://localhost:9001 \
--api-key EC9puzE6crcRm7buAF1S --model big400-body-insp-before-cleaning-7cr8v/8 \
--th 0.01 --trigger-file /tmp/body_go --result-file /tmp/body_res" \
"$LOG_DIR/body_client.log"

# Start Lip Client
start_terminal "Lip Client" \
"python /home/nvidia/big400_bottle_inspection/inference_clients/lip_client.py \
--daemon --cam-index 2 --api-url http://localhost:9001 \
--api-key EC9puzE6crcRm7buAF1S --model big400-lip-insp-before-cleaning-da2vm/9 \
--th 0.05 --trigger-file /tmp/lip_go --result-file /tmp/lip_res" \
"$LOG_DIR/lip_client.log"

# Start Body_CL Client
start_terminal "Body_CL Client" \
"python /home/nvidia/big400_bottle_inspection/inference_clients/body_cl_client.py \
--daemon --cam-index 1 --api-url http://localhost:9001 \
--api-key EC9puzE6crcRm7buAF1S --model big400-lip-insp-before-cleaning-da2vm/8 \
--th 0.8 --trigger-file /tmp/bodycl_go --result-file /tmp/bodycl_res" \
"$LOG_DIR/bodycl_client.log"

# Start Modbus Orchestrator
start_terminal "Modbus Orchestrator" \
"python /home/nvidia/big400_bottle_inspection/orchestrator_modbus_deamon_hr.py \
--bind-ip 0.0.0.0 --bind-port 5020 --timeout 7 --ack-timeout 2 \
--lip-trigger /tmp/lip_go --lip-result /tmp/lip_res \
--body-trigger /tmp/body_go --body-result /tmp/body_res \
--bodycl-trigger /tmp/bodycl_go --bodycl-result /tmp/bodycl_res" \
"$LOG_DIR/modbus_orchestrator.log"

