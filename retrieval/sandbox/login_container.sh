#! /bin/bash

#set -x
shopt -s lastpipe

enroot-exec(){ exec xargs -0 -a /proc/$1/environ bash -c 'env -i "${@:$(($0+1))}" nsenter --preserve-credentials -U -m -t "${@:1:$0}"' $# "$@"; }

scontrol listpids | awk '(NR>1 && $3 != "4294967295" && $4 != "-"){print $2"."$3" "$1}' | sort -n | while read -r id pid; do
    if lsns -o PID | grep -q "${pid}"; then
        jobs+=("[${id}] $(lsns -p "${pid}" -t user -o command | tail -n +2)")
        pids+=("${pid}")
    fi
done

# enroot exec "${pids[$((0))]}" bash
# enroot exec "${pids[$((0))]}" python
enroot exec "${pids[$((0))]}" "python $1"
# enroot exec "${pids[$((0))]}" "python --version"
# ls
# python

# eof
