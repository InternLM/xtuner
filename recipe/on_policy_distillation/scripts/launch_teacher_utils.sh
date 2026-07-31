#!/usr/bin/env bash

# Source-only helpers for launching, waiting for, and stopping OPD Teacher servers.

OPD_RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

TEACHER_NAMES=()
TEACHER_ENDPOINTS=()
TEACHER_HEALTH_URLS=()
TEACHER_MODEL_INFO_URLS=()
TEACHER_LOG_FILES=()
TEACHER_PIDS=()
STUDENT_CUDA_VISIBLE_DEVICES=
XTUNER_RL_NUM_WORKERS=

start_single_teacher_server() {
    _start_teacher_servers "$1" "$2" "$3" "1"
}

start_teacher_servers() {
    _start_teacher_servers "$1" "$2" "$3" ""
}

wait_for_teacher_servers() {
    local startup_timeout_s=$1
    local deadline=$((SECONDS + startup_timeout_s))
    local teacher_index
    local pid
    local log_file
    local -a teacher_ready=()
    local -a health_check_pids=()
    local -a pending_names=()

    if (( ${#TEACHER_NAMES[@]} == 0 )); then
        echo "No Teacher servers have been configured." >&2
        return 1
    fi

    for teacher_index in "${!TEACHER_NAMES[@]}"; do
        teacher_ready[teacher_index]=0
    done

    while (( SECONDS < deadline )); do
        for teacher_index in "${!TEACHER_NAMES[@]}"; do
            pid="${TEACHER_PIDS[teacher_index]}"
            log_file="${TEACHER_LOG_FILES[teacher_index]}"
            if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
                echo "Teacher ${TEACHER_NAMES[teacher_index]} exited before becoming ready." >&2
                if [[ -n "${log_file}" ]]; then
                    tail -n 50 "${log_file}" >&2 || true
                fi
                return 1
            fi
        done

        health_check_pids=()
        for teacher_index in "${!TEACHER_NAMES[@]}"; do
            if (( teacher_ready[teacher_index] )); then
                continue
            fi
            curl -sf --max-time 2 \
                "${TEACHER_HEALTH_URLS[teacher_index]}" \
                >/dev/null 2>&1 &
            health_check_pids[teacher_index]="$!"
        done

        for teacher_index in "${!health_check_pids[@]}"; do
            if wait "${health_check_pids[teacher_index]}"; then
                teacher_ready[teacher_index]=1
                echo "Teacher ${TEACHER_NAMES[teacher_index]} is ready at ${TEACHER_ENDPOINTS[teacher_index]}"
            fi
        done

        pending_names=()
        for teacher_index in "${!TEACHER_NAMES[@]}"; do
            if (( ! teacher_ready[teacher_index] )); then
                pending_names+=("${TEACHER_NAMES[teacher_index]}")
            fi
        done
        if (( ${#pending_names[@]} == 0 )); then
            return 0
        fi

        echo "Waiting for teachers: ${pending_names[*]}"
        sleep 5
    done

    for teacher_index in "${!TEACHER_NAMES[@]}"; do
        if (( teacher_ready[teacher_index] )); then
            continue
        fi
        echo "Teacher ${TEACHER_NAMES[teacher_index]} did not become ready within ${startup_timeout_s}s." >&2
        log_file="${TEACHER_LOG_FILES[teacher_index]}"
        if [[ -n "${log_file}" ]]; then
            tail -n 50 "${log_file}" >&2 || true
        fi
    done
    return 1
}

stop_teacher_servers() {
    local attempt
    local any_alive
    local pid

    for pid in "${TEACHER_PIDS[@]}"; do
        if [[ -n "${pid}" ]]; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        fi
    done

    for ((attempt = 0; attempt < 30; attempt++)); do
        any_alive=0
        for pid in "${TEACHER_PIDS[@]}"; do
            if [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        if (( ! any_alive )); then
            break
        fi
        sleep 1
    done

    for pid in "${TEACHER_PIDS[@]}"; do
        if [[ -n "${pid}" ]]; then
            if kill -0 -- "-${pid}" 2>/dev/null; then
                kill -KILL -- "-${pid}" 2>/dev/null || true
            fi
            wait "${pid}" 2>/dev/null || true
        fi
    done

    TEACHER_PIDS=()
}

_start_teacher_servers() {
    local config_file=$1
    local backend=$2
    local work_dir=$3
    local expected_teacher_count=$4
    local teacher_count
    local teacher_index
    local teacher_field_offset=4
    local teacher_name
    local teacher_target_node_rank
    local teacher_local_devices
    local teacher_endpoint
    local teacher_command_arg_count
    local teacher_log_file
    local device
    local student_local_num_workers
    local -a teacher_fields=()
    local -a teacher_command=()

    TEACHER_NAMES=()
    TEACHER_ENDPOINTS=()
    TEACHER_HEALTH_URLS=()
    TEACHER_MODEL_INFO_URLS=()
    TEACHER_LOG_FILES=()
    TEACHER_PIDS=()
    STUDENT_CUDA_VISIBLE_DEVICES=
    XTUNER_RL_NUM_WORKERS=
    mkdir -p "${work_dir}"

    # Command builder output:
    # count, endpoint map JSON, Student worker count, current-node Student
    # worker count, then repeated name, target node rank, local devices,
    # endpoint, health URL, model-info URL,
    # command-argument count, and command argv.
    unset XTUNER_OPD_TEACHER_ENDPOINTS_JSON
    mapfile -d "" -t teacher_fields < <(
        python "${OPD_RECIPE_DIR}/build_teacher_server_commands.py" \
            "${config_file}" "${backend}"
    )
    export XTUNER_OPD_TEACHER_ENDPOINTS_JSON="${teacher_fields[1]}"
    export XTUNER_RL_NUM_WORKERS="${teacher_fields[2]}"
    student_local_num_workers=${teacher_fields[3]}
    STUDENT_CUDA_VISIBLE_DEVICES=
    for ((device = 0; device < student_local_num_workers; device++)); do
        STUDENT_CUDA_VISIBLE_DEVICES+="${STUDENT_CUDA_VISIBLE_DEVICES:+,}${device}"
    done
    export STUDENT_CUDA_VISIBLE_DEVICES

    teacher_count=${teacher_fields[0]}
    if [[ -n "${expected_teacher_count}" ]] && (( teacher_count != expected_teacher_count )); then
        echo "Expected ${expected_teacher_count} Teacher, got ${teacher_count}." >&2
        return 1
    fi
    if (( teacher_count == 0 )); then
        echo "OPD config does not contain any Teachers: ${config_file}" >&2
        return 1
    fi

    echo "Teacher backend: ${backend}"
    echo "Student workers: ${XTUNER_RL_NUM_WORKERS}"
    echo "Student local GPUs on node ${NODE_RANK}: ${STUDENT_CUDA_VISIBLE_DEVICES:-none}"
    for ((teacher_index = 0; teacher_index < teacher_count; teacher_index++)); do
        teacher_name=${teacher_fields[teacher_field_offset]}
        teacher_target_node_rank=${teacher_fields[teacher_field_offset + 1]}
        teacher_local_devices=${teacher_fields[teacher_field_offset + 2]}
        teacher_endpoint=${teacher_fields[teacher_field_offset + 3]}
        teacher_command_arg_count=${teacher_fields[teacher_field_offset + 6]}

        TEACHER_NAMES+=("${teacher_name}")
        TEACHER_ENDPOINTS+=("${teacher_endpoint}")
        TEACHER_HEALTH_URLS+=("${teacher_fields[teacher_field_offset + 4]}")
        TEACHER_MODEL_INFO_URLS+=("${teacher_fields[teacher_field_offset + 5]}")

        teacher_field_offset=$((teacher_field_offset + 7))
        teacher_command=(
            "${teacher_fields[@]:teacher_field_offset:teacher_command_arg_count}"
        )
        teacher_field_offset=$((teacher_field_offset + teacher_command_arg_count))

        if (( teacher_command_arg_count == 0 )); then
            if [[ -n "${expected_teacher_count}" ]]; then
                echo "Teacher ${teacher_name} must define launch_config for local startup." >&2
                return 1
            fi
            echo "Using externally managed Teacher ${teacher_name} at ${teacher_endpoint}"
            TEACHER_LOG_FILES+=("")
            TEACHER_PIDS+=("")
            continue
        fi

        if (( NODE_RANK != teacher_target_node_rank )); then
            echo \
                "Teacher ${teacher_name} is assigned to node ${teacher_target_node_rank};" \
                "skipping node ${NODE_RANK}."
            TEACHER_LOG_FILES+=("")
            TEACHER_PIDS+=("")
            continue
        fi

        if [[ -n "${expected_teacher_count}" ]]; then
            teacher_log_file="${work_dir}/teacher.log"
        else
            teacher_log_file="${work_dir}/teacher_${teacher_index}_node_${NODE_RANK}.log"
        fi
        TEACHER_LOG_FILES+=("${teacher_log_file}")

        echo "Starting Teacher ${teacher_name}"
        echo "Teacher node: ${NODE_RANK}"
        echo "Teacher local GPUs: ${teacher_local_devices}"
        echo "Teacher endpoint: ${teacher_endpoint}"
        echo "Teacher log: ${teacher_log_file}"

        setsid env \
            PYTHONUNBUFFERED=1 \
            CUDA_VISIBLE_DEVICES="${teacher_local_devices}" \
            "${teacher_command[@]}" \
            >"${teacher_log_file}" 2>&1 &
        TEACHER_PIDS+=("$!")
    done

    if (( teacher_field_offset != ${#teacher_fields[@]} )); then
        echo "Teacher command builder returned malformed records." >&2
        return 1
    fi
}
